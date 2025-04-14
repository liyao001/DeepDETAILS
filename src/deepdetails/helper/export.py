import os
import shutil
import h5py
import numpy as np
from deepdetails.data import parse_regions


STRAND_LABELS = ("pl", "mn")
STRAND_COEFF = (1, -1)


def preds_to_bg_star(pred_h5: str, cluster_idx: int, strand_idx: int, save_to: str, min_abs_val: float = 10e-3,
                     bins: int = 0):
    """Convert predictions to bedGraph files

    Parameters
    ----------
    pred_h5 : str
        Prediction hdf5 file
    cluster_idx : int
        Cluster / cell type index
    strand_idx : int
        Strand index
    save_to : str
        Save outputs to this directory
    min_abs_val : float
        Minimum absolute value for a locus to be reported
    bins : int
        {out_binning}

    Returns
    -------
    n_strands : int
        Number of strands that have predictions
    """
    with h5py.File(pred_h5, "r") as f:
        regions = parse_regions(f)
        cluster_preds, chroms, starts, ends = f["preds"], regions[0], regions[1], regions[2]

        output_name = f"C{cluster_idx}.{STRAND_LABELS[strand_idx]}.bg"
        fh = open(os.path.join(save_to, output_name), "w")
        for r_i in range(regions.shape[0]):
            sample_pred = cluster_preds[cluster_idx, r_i, strand_idx, :]
            coords = np.arange(starts[r_i], ends[r_i])
            coords1 = coords + 1
            if bins > 0:
                coords = coords.reshape(-1, bins).min(axis=1)
                coords1 = coords1.reshape(-1, bins).max(axis=1)
                sample_pred = sample_pred.reshape(-1, bins).mean(axis=1)

            fh.writelines(
                [f"{chroms[r_i]}\t{coords[i]}\t{coords1[i]}\t{sample_pred[i]}\n" for i in
                 np.where(np.abs(sample_pred) > min_abs_val)[0]]
            )
        fh.close()
    return output_name


def bg_to_bw_core(bg_file: str, prefix: str, chrom_size: str, coef: int = 1, skip_sort_merge: bool = False) -> str:
    """Convert bedGraph files to bigWigs

    Parameters
    ----------
    bg_file : str
        A bedGraph files
    prefix : str
        Output prefix
    chrom_size : str

    coef : int
        Coefficient to be applied to the signal values.
        1 for the positive strand, -1 for the negative strand
    skip_sort_merge : bool
        If the input bg_file is sorted and there are no overlapping regions (or they get merged before),
        set this as True to speed up computation.

    Returns
    -------
    dest_bw : str
        Destination bigWig file
    """
    _house_keeping = []
    _house_keeping.append(bg_file)
    dependent_tools = ("sort", "bedtools", "bedGraphToBigWig")
    for tool in dependent_tools:
        full_path = shutil.which(tool)
        if full_path is None:
            raise RuntimeError(f"Required tool {tool} is not callable, please make sure it's in your PATH.")

    dest_bg = f"{prefix}.bedGraph"
    _house_keeping.append(dest_bg)
    if not skip_sort_merge:
        cmd = f"sort -T . -k1,1 -k2,2n {bg_file} | "
        cmd += "bedtools merge -i stdin -d -1 -c 4 -o mean | awk 'BEGIN{OFS=\"\\t\";FS=\"\\t\"}"
        cmd += f"{{print $1,$2,$3,$4*{coef}}}' > {dest_bg}"
        print(cmd)
        os.system(cmd)
    else:
        cmd = "awk 'BEGIN{OFS=\"\\t\";FS=\"\\t\"} "
        cmd += f"{{print $1,$2,$3,$4*{coef}}}' "
        cmd += f"{bg_file} > {dest_bg}"
        print(cmd)
        os.system(cmd)

    dest_bw = f"{prefix}.bw"
    cmd = f"bedGraphToBigWig {dest_bg} {chrom_size} {dest_bw}"
    print(cmd)
    os.system(cmd)

    # clean up
    for f in _house_keeping: os.system(f"rm {f}")
    return dest_bw
