import logging
import os
import pyBigWig
import pybedtools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Iterable
from scipy.stats import ttest_ind, false_discovery_control
from sklearn.svm import NuSVR
from deepdetails.par_description import PARAM_DESC

logger = logging.getLogger("Preflight Check")
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)


def load_bulk_signal(region: pd.DataFrame, bulk_bws: tuple) -> pd.Series:
    """
    Load signal from bulk bw file

    Parameters
    ----------
    region : pd.DataFrame

    bulk_bws : tuple


    Returns
    -------
    signals : pd.Series
        Strand-aggregated bulk signal
    """
    bw_objs = [pyBigWig.open(bbw) for bbw in bulk_bws if bbw is not None and os.path.exists(bbw)]

    def row_atom_func(row):
        try:
            return np.abs([v if v is not None else 0. for v in
                           [bw.stats(row[0], row[1], row[2], type="sum", exact=True)[0] for bw in bw_objs]]).sum()
        except:
            return 0.

    values = region.apply(row_atom_func, axis=1)
    for bwo in bw_objs: bwo.close()
    return values


def build_aggregated_counts_table(fragment_files_dict: dict, regions: pybedtools.BedTool,
                                  barcodes: pd.DataFrame, n_aggs: int = 5, min_cells_required: int = 20
                                  ) -> tuple[np.ndarray, tuple]:
    """
    Build aggregated counts table

    Parameters
    ----------
    fragment_files_dict : dict
        key: name of the corresponding cluster
        value: path to the fragment file
    regions : pybedtools.BedTool
        peak regions
    barcodes : pd.DataFrame
        barcode (column 0) and its corresponding cluster (column 1)
    n_aggs : int
        Number of aggregations
    min_cells_required : int


    Returns
    -------
    counts_mat : np.ndarray
        Aggregated counts table
    ordered_groups : tuple
        Column names for the counts_mat
    """
    logger.info("Grouping barcodes")
    barcode_groups = barcodes.groupby(1)
    ordered_groups = list(sorted(fragment_files_dict.keys()))
    n_cell_types = len(ordered_groups)
    n_regions = len(regions)
    counts_mat = np.zeros((n_cell_types, n_aggs, n_regions))
    group_sizes = barcode_groups.size()

    to_be_removed = set()
    for i in range(n_aggs):
        logger.info(f"Generating aggregated profiles: {i + 1} / {n_aggs}")
        for ct, size in group_sizes.items():
            idx = ordered_groups.index(ct)
            if size < min_cells_required:
                to_be_removed.add((idx, ct))
            fragments = pd.read_csv(fragment_files_dict[ct], sep="\t", header=None)
            # sample cells
            sample_size = size // 2
            sampled_cells = set(np.random.choice(barcode_groups.get_group(ct)[0], sample_size, replace=False))
            sampled_frags = fragments.loc[fragments[3].isin(sampled_cells)]
            tmp_frags = fragment_files_dict[ct] + ".tmp"
            sampled_frags.to_csv(tmp_frags, sep="\t", header=False, index=False)
            logger.info(f"Sampled {sample_size} cells and {sampled_frags.shape[0]} fragments for {ct}")
            # get aggregated counts
            logger.info(f"Building counts table for {ct}")
            cov = regions.coverage(pybedtools.BedTool(tmp_frags)).to_dataframe(disable_auto_names=True, header=None)[3]
            # normalize by depth
            counts_mat[idx, i, :] = cov * (1_000000. / sampled_frags.shape[0])
            os.remove(tmp_frags)
    if len(to_be_removed) > 0:
        all_cts = list(range(n_cell_types))
        for ci, ct in to_be_removed:
            all_cts.remove(ci)
            ordered_groups.remove(ct)

        counts_mat = counts_mat[all_cts, :, :]
    return counts_mat, tuple(ordered_groups)


def plot_signature_mat(sig_df: pd.DataFrame, save_to: str):
    """
    Plot signature matrix

    Parameters
    ----------
    sig_df : pd.DataFrame
        Signature matrix
    save_to : str
        Save plot to this folder

    Returns
    -------

    """
    fig, ax = plt.subplots()
    im = ax.imshow(sig_df, aspect="auto", interpolation="none")
    ax.set_xticks(np.arange(sig_df.shape[1]))
    ax.set_xticklabels(sig_df.columns, rotation=45, ha="right")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(save_to, "signatures.png"))
    plt.close()


def get_signature_distribution(
        counts_mat: np.ndarray, regions_df: pd.DataFrame, counts_mat_labels: tuple,
        save_to: str, qval_cutoff: float = 0.01, fc_cutoff: float = 2., max_top_n: int = 1000
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Find signature candidates

    Parameters
    ----------
    counts_mat : np.ndarray
        Aggregated counts matrix. Shape: n_ct, n_agg, n_regions
    regions_df : pd.DataFrame
        Regions dataframe
    counts_mat_labels : tuple
        Column names for the counts_mat
    save_to : str
        Save outputs to this folder
    qval_cutoff : float
        Q-value cutoff for a region to be considered a signature
    fc_cutoff : float
        Fold-change cutoff for a region to be considered a signature
    max_top_n : int
        Maximum number of signature candidates to consider for each cluster

    Returns
    -------
    raw_array : tuple[np.ndarray, np.ndarray]
        Signature mat, shape: n_ct, n_agg, max_top_n * n_ct
        bulk vector, shape: max_top_n * n_ct
    qnorm_array : tuple[np.ndarray, np.ndarray]
        Quantile-normalized signature and bulk mats
    """
    n_cell_types = len(counts_mat_labels)

    all_candidates = {}
    for c_i, c in enumerate(counts_mat_labels):
        logger.info(f"Identifying open chromatin signatures for {c}")
        other_cols = list(range(n_cell_types))
        other_cols.remove(c_i)
        g1 = counts_mat[[c_i], :, :].reshape(-1, counts_mat.shape[-1])
        g2 = counts_mat[other_cols, :, :].reshape(-1, counts_mat.shape[-1])
        # p/q-value
        pvals = np.nan_to_num(ttest_ind(g1, g2, axis=0, equal_var=False, alternative="greater").pvalue, nan=1.)
        regions_df[f"{c}_pval"] = pvals
        qvals = false_discovery_control(pvals, method="bh")
        regions_df[f"{c}_qval"] = qvals
        # fold change
        regions_df[f"{c}_fc"] = g1.mean(axis=0) / (g2.mean(axis=0) + 1e-8)

        c_candidates = regions_df.loc[
            (regions_df[f"{c}_qval"] < qval_cutoff) & (regions_df[f"{c}_fc"] > fc_cutoff),
            (f"{c}_fc", "bulk")
        ].sort_values(by=f"{c}_fc", ascending=False)
        if c_candidates.shape[0] == 0:  # cannot find signatures
            logger.warning(f"Cannot identify any signatures for cluster {c}")
        all_candidates[c] = c_candidates

    top_n = min(max_top_n, min([len(c) for c in all_candidates.values() if len(c) > 0]))
    logger.info(
        f"For each cell type/cluster, the top {top_n} differentially accessible regions will serve as signatures")
    candidate_indexes = []

    for c, candidates in all_candidates.items():
        sub = candidates[:top_n]
        candidate_indexes.extend(sub.index.to_list())

    counts_mean = counts_mat.mean(axis=1)
    signature_idx = list(dict.fromkeys(candidate_indexes))
    final_sig_mat = np.log1p(counts_mean[:, signature_idx]).T
    region_names = regions_df[0] + ":" + regions_df[1].map(str) + "-" + regions_df[2].map(str)
    final_sig = pd.DataFrame(final_sig_mat, index=region_names[signature_idx], columns=list(counts_mat_labels))
    final_sig.to_csv(os.path.join(save_to, "signatures.csv"))

    sig_regions = regions_df.copy()
    qn_sig_regions = regions_df.copy()
    for c_i, c in enumerate(counts_mat_labels):
        sig_regions[c] = counts_mean[c_i, :]
        qn_sig_regions[c] = counts_mean[c_i, :]
    sig_regions = sig_regions.loc[signature_idx].copy().reset_index(drop=True)

    ref_ranks = qn_sig_regions["bulk"].rank(method="min")
    ref_values = qn_sig_regions["bulk"]
    sorted_indices = np.argsort(ref_ranks)
    sorted_ref_ranks = ref_ranks[sorted_indices]
    sorted_ref_values = ref_values[sorted_indices]
    for c in counts_mat_labels:
        col_ranks = qn_sig_regions[c].rank(method="min")
        qn_sig_regions[c] = np.interp(col_ranks, sorted_ref_ranks, sorted_ref_values, left=0)
    qn_sig_regions = qn_sig_regions.loc[signature_idx].copy().reset_index(drop=True)

    plot_signature_mat(final_sig, save_to)
    col_lst = list(counts_mat_labels)
    return (sig_regions[col_lst].values, sig_regions["bulk"].values), (
        qn_sig_regions[col_lst].values, qn_sig_regions["bulk"].values)


def frac_based_diagnose(A: np.ndarray, b: np.ndarray, cluster_labels: tuple,
                        nu: float = 0.85, detection_cutoff: float = 0.04) -> set:
    """

    Parameters
    ----------
    A : np.ndarray
        Signature signal values
    b : np.ndarray
        Bulk signal values
    cluster_labels : tuple

    nu : Optional[float]

    detection_cutoff : Optional[float]

    Returns
    -------
    to_exclude : set

    """
    to_exclude = set()

    # nu-SVR
    regr = NuSVR(C=1.0, nu=nu, kernel="linear")
    res = regr.fit(A, b)
    nv_coefs = np.clip(res.coef_.flatten(), a_min=0., a_max=None)
    normed_coef = nv_coefs / np.sum(nv_coefs)
    logger.info(f"Estimation from v-SVR: {normed_coef} ({res.score(A, b)})")

    final_est = (nv_coefs / np.sum(nv_coefs)).flatten()
    for c, f in zip(cluster_labels, final_est):
        if f < detection_cutoff:
            logger.warning(
                f"v-SVR: Cell type / cluster {c} may not exist in the bulk library as the estimated fraction is {f}")
            logger.warning(f"Cell type / cluster {c} will be excluded from downstream analysis.")
            to_exclude.add(c)
    return to_exclude


def preflight_check(fragment_files: dict, barcodes: pd.DataFrame, regions: pd.DataFrame, bulks: tuple[str, str],
                    n_aggs: int = 5, max_top_n: int = 1000, qval_cutoff: float = 0.01, fc_cutoff: float = 2.,
                    preflight_cutoff: float = 0.035, nu: float = 0.85, min_cells_required: int = 20,
                    use_qnorm: bool = False, save_to: str = ".") -> Iterable:
    """
    Run preflight check to find clusters that may not exist in the bulk library

    Parameters
    ----------
    fragment_files : dict

    barcodes : pd.DataFrame

    regions : pd.DataFrame

    bulks : tuple[str, str]

    n_aggs : int
        {n_aggs}
    max_top_n : int
        {max_top_n}
    qval_cutoff : float
        {qval_cutoff}
    fc_cutoff : float
        {fc_cutoff}
    preflight_cutoff : float
        {preflight_cutoff}
    nu : float
        {preflight_nu}
    min_cells_required : int
        {min_cells_required}
    use_qnorm : bool
        {use_qnorm}
    save_to : str


    Returns
    -------
    to_be_excluded : Iterable
        Cell types to be excluded from downstream analysis
    """.format(**PARAM_DESC)
    if regions.shape[1] > 3:
        regions = regions[[0, 1, 2]]
    regions = regions.drop_duplicates().copy().reset_index(drop=True)
    regions_bed = pybedtools.BedTool.from_dataframe(regions)
    logger.info("Loading signal from the bulk library")
    regions_bulk = load_bulk_signal(regions, bulks)
    ext_regions = regions.copy()
    ext_regions["bulk"] = regions_bulk

    counts_mat, cluster_labels = build_aggregated_counts_table(
        fragment_files, regions_bed, barcodes, n_aggs, min_cells_required)
    logger.info(f"Counts table ready, shape: {counts_mat.shape}")
    to_be_excluded = set(fragment_files.keys()).difference(set(cluster_labels))
    if len(to_be_excluded) > 0:
        logger.info(f"Clusters {to_be_excluded} will be excluded because of low fragment counts in the reference")

    # find signatures
    (A, b), (Aq, bq) = get_signature_distribution(
        counts_mat, ext_regions, cluster_labels, save_to, qval_cutoff, fc_cutoff, max_top_n)

    # Preflight check
    if use_qnorm:
        to_be_excluded = to_be_excluded.union(
            frac_based_diagnose(Aq, bq, cluster_labels, nu, detection_cutoff=preflight_cutoff))
    else:
        to_be_excluded = to_be_excluded.union(
            frac_based_diagnose(A, b, cluster_labels, nu, detection_cutoff=preflight_cutoff))
    return to_be_excluded
