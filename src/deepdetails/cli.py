"""DeepDETAILS: Deep-learning-based DEconvolution of Tissue profiles with Accurate
Interpretation of Locus-specific Signals

Run `deepdetails <function> --help` for details on each subcommand's arguments.
"""
import argparse
import os
from glob import glob
from typing import Sequence

from deepdetails.__about__ import __version__
from deepdetails.helper.argtypes import (
    DevicesAction,
    alpha_as_fraction,
    created_dir,
    existing_dir,
    existing_file,
)
from deepdetails.helper.utils import check_update, require_external_binaries
from deepdetails.par_description import PARAM_DESC, RescalingMode


class _Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Preserve line breaks in the parser's description/epilog while still
    appending "(default: ...)" to each argument's help text."""


def _args_to_kwargs(args: argparse.Namespace, drop: set[str] | None = None) -> dict:
    drop = {"function", "handler", *(drop or set())}
    return {k: v for k, v in vars(args).items() if k not in drop}


def _run_deconv(args: argparse.Namespace):
    from deepdetails.protocols import deconv

    deconv(**_args_to_kwargs(args))


def _run_prep_data(args: argparse.Namespace, parser: argparse.ArgumentParser):
    from deepdetails.protocols import prepare_dataset

    if args.fragments and args.barcodes is None:
        parser.error("--fragments requires --barcodes to be specified")
    prepare_dataset(**_args_to_kwargs(args))


def _run_export_pred(args: argparse.Namespace):
    from deepdetails.model.wrapper import DeepDETAILS
    from deepdetails.protocols import export_results

    export_results(model=DeepDETAILS, **_args_to_kwargs(args, {"hide_progress_bar", "version"}))


def _run_export_gw_pred(args: argparse.Namespace):
    from deepdetails.model.wrapper import DeepDETAILS
    from deepdetails.protocols import export_wg_results

    export_wg_results(model=DeepDETAILS, **_args_to_kwargs(args, {"hide_progress_bar", "version"}))


def _run_build_bw(args: argparse.Namespace):
    from deepdetails.protocols import pred_to_bw

    pred_to_bw(**_args_to_kwargs(args))


def _run_merge_preds(args: argparse.Namespace):
    from deepdetails.protocols import merge_rep_preds

    if args.pred_dir:
        pred_files = sorted(glob(os.path.join(args.pred_dir, "*predictions.h5")))
    else:
        pred_files = args.preds
    merge_rep_preds(
        in_pred_files=pred_files, save_to=args.save_to,
        keep_old=args.keep_old_preds, quiet=args.quiet,
    )


def _run_attr(args: argparse.Namespace):
    from deepdetails.protocols import export_attr

    export_attr(**_args_to_kwargs(args, {"version", "study_name", "num_workers"}))


def _general_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("General")
    group.add_argument("--study-name", default="DeepDETAILS", type=str, help=PARAM_DESC["study_name"])
    group.add_argument("--save-to", default=".", type=created_dir, help=PARAM_DESC["save_to"])
    group.add_argument("--num-workers", help=PARAM_DESC["num_workers"], type=int, default=16)
    group.add_argument("--batch-size", help=PARAM_DESC["batch_size"],
                       type=int, default=32)
    group.add_argument("--hide-progress-bar", action="store_true", help=PARAM_DESC["hide_progress_bar"])
    group.add_argument("--version", action="version", version=__version__)


def _training_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Training")
    group.add_argument("--dataset", required=True, type=existing_dir, help=PARAM_DESC["dataset"])
    group.add_argument("--chrom-cv", action="store_true", help=PARAM_DESC["chrom_cv"])
    group.add_argument("--chromosomal-validation", "--cv", dest="cv", default=("chr22",),
                       help=PARAM_DESC["chromosomal_validation"], nargs="*")
    group.add_argument("--chromosomal-testing", "--ct", dest="ct", default=("chr19",),
                       help=PARAM_DESC["chromosomal_testing"], nargs="*")
    group.add_argument("--accelerator", type=str, default="auto",
                       choices=("gpu", "tpu", "auto", "cpu", "ipu"), help=PARAM_DESC["accelerator"])
    group.add_argument("--devices", help=PARAM_DESC["devices"],
                       nargs="+", action=DevicesAction, default=[0])
    group.add_argument("--earlystop-patience", help=PARAM_DESC["earlystop_patience"],
                       type=int, default=2)
    group.add_argument("--min-delta", help=PARAM_DESC["min_delta"],
                       type=float, default=0.0001)
    group.add_argument("--max-epochs", help=PARAM_DESC["max_epochs"],
                       type=int, default=50)
    group.add_argument("--save-top-k-model", help=PARAM_DESC["save_top_k_model"],
                       default=1, type=int)
    group.add_argument("--resume", "--resume-from-ckpt", dest="resume_from_ckpt",
                       help=PARAM_DESC["resume_from_ckpt"], type=str)
    # for backward compatibility
    g = group.add_mutually_exclusive_group()
    g.add_argument("--save-preds", action="store_true", default=True, help=PARAM_DESC["save_preds"])
    g.add_argument("--no-preds", action="store_false", dest="save_preds", help=PARAM_DESC["no_preds"])
    group.add_argument("--alpha", "--redundancy-loss-coef",
                       help=f"{PARAM_DESC['redundancy_loss_coef']} (0-100 scale; divided by 100 internally)",
                       type=alpha_as_fraction, default="1", dest="redundancy_loss_coef",
                       metavar="{0..100}")
    group.add_argument("--prior-loss-coef", help=PARAM_DESC["prior_loss_coef"],
                       type=float, default=1.0)
    group.add_argument("--gamma", help=PARAM_DESC["gamma"],
                       type=float, default=1e-8)
    group.add_argument("--learning-rate", help=PARAM_DESC["learning_rate"],
                       type=float, default=1e-3)
    group.add_argument("--lr-step-size", help=PARAM_DESC["lr_step_size"],
                       type=int, default=1)
    group.add_argument("--lr-gamma", help=PARAM_DESC["lr_gamma"],
                       type=float, default=0.1)
    group.add_argument("--betas", help=PARAM_DESC["betas"],
                       type=float, default=(0.9, 0.999), nargs=2)
    group.add_argument("--model-summary-depth", help=PARAM_DESC["max_depth"],
                       type=int, default=6)
    group.add_argument("--max-retry", help=PARAM_DESC["max_retry"],
                       type=int, default=3)
    group.add_argument("--rescaling-mode", dest="rescaling_mode", help=PARAM_DESC["rescaling_mode"],
                       choices=RescalingMode.values(), default=RescalingMode.COUNTS, type=RescalingMode.parse)
    # for backward compatibility
    g = group.add_mutually_exclusive_group()
    g.add_argument("--all-regions", action="store_true", help=PARAM_DESC["all_regions"], default=True)
    g.add_argument("--peak-only", action="store_false", dest="all_regions", help=PARAM_DESC["peak_only"])
    group.add_argument("--test-all-regions", dest="test_pos_only", action="store_false",
                        help=PARAM_DESC["test_pos_only"])
    group.add_argument("-v", "--version-tag", dest="version", help=PARAM_DESC["wandb_version"],
                       type=str, default="")
    group.add_argument("--loads-trunc", required=False, type=int, help=PARAM_DESC["loads_trunc"])


def _model_conf_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Model Configuration")
    group.add_argument("--profile-shrinkage", help=PARAM_DESC["profile_shrinkage"],
                       type=int, default=8, required=False)
    group.add_argument("--filters", help=PARAM_DESC["filters"], type=int, default=512)
    group.add_argument("--head-mlp-layers", dest="head_layers",
                       type=int, default=3, help=PARAM_DESC["head_mlp_layers"])
    group.add_argument("--gru-layers", help=PARAM_DESC["gru_layers"],
                       type=int, default=2)
    group.add_argument("--gru-dropout", help=PARAM_DESC["gru_dropout"],
                       type=float, default=0.1)
    group.add_argument("--scale-function-placement", choices=("early", "late", "late-ch", "disable"),
                       help=PARAM_DESC["scale_function_placement"], default="late-ch")
    group.add_argument("--seq", dest="seq_only", action="store_true", required=False,
                       help=PARAM_DESC["seq_only"])
    group.add_argument("--n-times-more-embeddings", help=PARAM_DESC["n_times_more_embeddings"],
                       type=int, default=2)


def _wandb_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("WandB")
    group.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    group.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    group.add_argument("--wandb-upload-model", action="store_true", required=False)


def _prep_dataset_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("")
    group.add_argument("--regions", nargs="+", required=True, type=existing_file,
                       help=PARAM_DESC["regions"])
    group.add_argument("--bulk-pl", required=True, type=existing_file, help=PARAM_DESC["bulk_pl"])
    group.add_argument("--bulk-mn", type=existing_file, help=PARAM_DESC["bulk_mn"])
    group.add_argument("--save-to", default=".", type=created_dir, help=PARAM_DESC["save_to"])
    group.add_argument("--genome-fa", type=existing_file, required=True, help=PARAM_DESC["genome_fa"])
    group.add_argument("--chrom-size", type=existing_file, required=True, help=PARAM_DESC["chrom_size"])
    group.add_argument("--seed", type=int, default=1234567, help=PARAM_DESC["seed"])
    group.add_argument("--ref-labels", nargs="*", type=str, help=PARAM_DESC["ref_labels"])
    group.add_argument("--ref-pls", nargs="*", type=existing_file, help=PARAM_DESC["ref_pls"])
    group.add_argument("--ref-mns", nargs="*", type=existing_file, help=PARAM_DESC["ref_mns"])
    group.add_argument("--background-sampling-ratio", help=PARAM_DESC["background_sampling_ratio"],
                       default=0., type=float)
    group.add_argument("--background-blacklist", type=existing_file, help=PARAM_DESC["background_blacklist"])
    group.add_argument("--final-regions", action="store_true", help=PARAM_DESC["final_regions"])
    group.add_argument("--keep-frags", action="store_true", help=PARAM_DESC["keep_frags"])
    group.add_argument("--disable-rpm", action="store_true", help=PARAM_DESC["disable_rpm"])
    group.add_argument("--memory-saving", action="store_true", help=PARAM_DESC["memory_saving"])
    group.add_argument("--collapse-missing-cell-types", action="store_true",
                       help=PARAM_DESC["collapse_missing_cell_types"])
    group.add_argument("--merge-overlap-peaks", help=PARAM_DESC["merge_overlap_peaks"],
                       type=int, default=0)
    group.add_argument("--target-sliding-sum", help=PARAM_DESC["target_sliding_sum"],
                       default=0, type=int)
    group0 = group.add_mutually_exclusive_group(required=True)
    group0.add_argument("--accessibility", nargs="+", type=existing_file, help=PARAM_DESC["accessibility"])
    group0.add_argument("--fragments", type=existing_file, help=PARAM_DESC["fragments"])
    group.add_argument("--barcodes", type=existing_file, help=PARAM_DESC["barcodes"])


def _preflight_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Preflight")
    group0 = group.add_mutually_exclusive_group(required=False)
    group0.add_argument("--skip-preflight", action="store_true", help=PARAM_DESC["skip_preflight"])
    group0.add_argument("--combine-cell-types", type=str, nargs="*", help=PARAM_DESC["combine_cell_types"])
    group.add_argument("--accessible-regions", dest="accessible_peaks", required=False,
                       type=existing_file, help=PARAM_DESC["accessible_peaks"])
    group.add_argument("--preflight-cutoff", help=PARAM_DESC["preflight_cutoff"],
                       default=0.035, type=float)
    group.add_argument("--nu", help=PARAM_DESC["preflight_nu"],
                       default=0.85, type=float)
    group.add_argument("--use-qnorm", action="store_true", help=PARAM_DESC["use_qnorm"])
    group.add_argument("--candidate-qval", help=PARAM_DESC["qval_cutoff"],
                       default=0.01, type=float)
    group.add_argument("--candidate-fc", help=PARAM_DESC["fc_cutoff"],
                       default=2., type=float)
    group.add_argument("--max-top-n", help=PARAM_DESC["max_top_n"],
                       type=int, default=1000)
    group.add_argument("--min-cells-required", help=PARAM_DESC["min_cells_required"],
                       type=int, default=20)
    group.add_argument("--n-aggs", help=PARAM_DESC["n_aggs"],
                       type=int, default=5)


def _export_pred_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Export HDF5")
    group.add_argument("--dataset", required=True, type=existing_dir, help=PARAM_DESC["dataset"])
    group.add_argument("-m", "--checkpoint", required=True, type=existing_file, help=PARAM_DESC["checkpoint"])
    group.add_argument("--rescaling-mode", dest="rescaling_mode", help=PARAM_DESC["rescaling_mode"],
                       choices=RescalingMode.values(), default=RescalingMode.COUNTS, type=RescalingMode.parse)
    group.add_argument("--all-regions", dest="pos_only", action="store_false",
                       help=PARAM_DESC["all_regions"])
    group.add_argument("--merge-strands", action="store_true",
                       help=PARAM_DESC["merge_strands"])
    group.add_argument("--device", help=PARAM_DESC["device"], type=str, default="cuda")
    group.add_argument("--loads-trunc", required=False, type=int, help=PARAM_DESC["loads_trunc"])


def _export_wg_pred_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Export whole-genome HDF5")
    group.add_argument("--genome-fa", dest="fa_file", required=True, type=existing_file,
                       help=PARAM_DESC["genome_fa"])
    group.add_argument("--regions-file", type=existing_file, help=PARAM_DESC["regions"])
    group.add_argument("--bulk-pl", dest="pl_bulk_bw_file", required=True, type=existing_file,
                       help=PARAM_DESC["bulk_pl"])
    group.add_argument("--bulk-mn", dest="mn_bulk_bw_file", type=existing_file, help=PARAM_DESC["bulk_mn"])
    group.add_argument("--accessibility", dest="acc_bw_files", nargs="+", type=existing_file,
                       help=PARAM_DESC["accessibility"])
    group.add_argument("--ref-pls", dest="pl_ct_bw_files", nargs="*", type=existing_file,
                       help=PARAM_DESC["ref_pls"])
    group.add_argument("--ref-mns", dest="mn_ct_bw_files", nargs="*", type=existing_file,
                       help=PARAM_DESC["ref_mns"])

    group.add_argument("--sc-norm", dest="sc_norm_file", type=existing_file, help=PARAM_DESC["sc_norm_file"])
    group.add_argument("--ref-labels", dest="cluster_names", nargs="*",
                       type=str, help=PARAM_DESC["ref_labels"])
    group.add_argument("--target-sliding-sum", help=PARAM_DESC["target_sliding_sum"],
                       default=0, type=int)
    group.add_argument("--dataset-mode", dest="is_training", help=PARAM_DESC["is_training"],
                       choices=(0, 1, 2, -1), default=-1, type=int)

    group.add_argument("-m", "--checkpoint", required=True, type=existing_file, help=PARAM_DESC["checkpoint"])
    group.add_argument("--rescaling-mode", dest="rescaling_mode", help=PARAM_DESC["rescaling_mode"],
                       choices=RescalingMode.values(), default=RescalingMode.COUNTS, type=RescalingMode.parse)
    group.add_argument("--all-regions", dest="pos_only", action="store_false",
                       help=PARAM_DESC["all_regions"])
    group.add_argument("--merge-strands", action="store_true",
                       help=PARAM_DESC["merge_strands"])
    group.add_argument("--use-bulk-constraint", action="store_true", required=False,
                       help=PARAM_DESC["use_bulk_constraint"])
    group.add_argument("--device", help=PARAM_DESC["device"], type=str, default="cuda")
    group.add_argument("--loads-trunc", required=False, type=int, help=PARAM_DESC["loads_trunc"])


def _build_bw_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Export BigWig")
    group.add_argument("-p", "--pred-file", required=True, type=existing_file, help=PARAM_DESC["pred_file"])
    group.add_argument("-s", "--save-to", default=".", type=created_dir, help=PARAM_DESC["save_to"])
    group.add_argument("-c", "--chrom-size", type=existing_file, required=True, help=PARAM_DESC["chrom_size"])
    group.add_argument("--min-abs-val", action="store", help=PARAM_DESC["min_abs_val"],
                       type=float, default=10e-3)
    group.add_argument("-f", "--fast", dest="skip_sort_merge", action="store_true",
                       help=PARAM_DESC["skip_sort_merge"])
    group.add_argument("--out-binning", dest="binning",
                       help=PARAM_DESC["out_binning"], type=int, default=0)
    group.add_argument("--num-workers", help=PARAM_DESC["num_workers"], type=int, default=16)


def _export_attr_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Export attribution tracks")
    group.add_argument("-m", "--checkpoint", required=True, type=existing_file, help=PARAM_DESC["checkpoint"])
    group.add_argument("--dataset", required=True, type=existing_dir, help=PARAM_DESC["dataset"])
    group.add_argument("-c", "--chrom-size", type=existing_file, required=False, help=PARAM_DESC["chrom_size"])
    group.add_argument("--device", help=PARAM_DESC["device"], type=str, default="cuda")


def _merge_preds(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Merge predictions from multiple replicate runs")
    g = group.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred-dir", type=existing_dir, help=PARAM_DESC["pred_dir"])
    g.add_argument("--preds", type=existing_file, nargs="+", help=PARAM_DESC["preds"])
    group.add_argument("--save-to", type=str,
                       required=True, help=PARAM_DESC["save_to"])
    group.add_argument("--keep-old-preds", action="store_true", help=PARAM_DESC["keep_old_preds"])
    group.add_argument("--quiet", action="store_true", help=PARAM_DESC["quiet"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)
    parser.add_argument("-v", "--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(title="Available functions", dest="function")

    # Subparser for deconvolution
    parser_deconv = subparsers.add_parser(
        "deconv", help="Using DeepDETAILS to deconvolve a bulk sample", formatter_class=_Formatter)
    _general_parser(parser_deconv)
    _training_parser(parser_deconv)
    _model_conf_parser(parser_deconv)
    _wandb_parser(parser_deconv)
    parser_deconv.set_defaults(handler=_run_deconv)

    # Subparser for dataset preparation
    parser_prep_data = subparsers.add_parser(
        "prep-data", help="Create a dataset for DeepDETAILS to deconvolve a bulk sample", formatter_class=_Formatter)
    _prep_dataset_parser(parser_prep_data)
    _preflight_parser(parser_prep_data)
    parser_prep_data.set_defaults(handler=lambda args: _run_prep_data(args, parser))

    # Subparser for export prediction hdf5
    parser_export_pred = subparsers.add_parser(
        "export-pred", help="Export predictions to a hdf5 file", formatter_class=_Formatter)
    _general_parser(parser_export_pred)
    _export_pred_parser(parser_export_pred)
    parser_export_pred.set_defaults(handler=_run_export_pred)

    parser_gw_export_pred = subparsers.add_parser(
        "export-gw-pred", help="Export whole-genome predictions to a hdf5 file", formatter_class=_Formatter)
    _general_parser(parser_gw_export_pred)
    _export_wg_pred_parser(parser_gw_export_pred)
    parser_gw_export_pred.set_defaults(handler=_run_export_gw_pred)

    # Subparser for export bigwigs
    parser_build_bw = subparsers.add_parser(
        "build-bw", help="Store predictions to BigWig files", formatter_class=_Formatter)
    _build_bw_parser(parser_build_bw)
    parser_build_bw.set_defaults(handler=_run_build_bw)

    # Subparser for merging predictions from replicate runs
    parser_merge_preds = subparsers.add_parser(
        "merge-preds", help="Merge predictions from multiple replicate runs", formatter_class=_Formatter)
    _merge_preds(parser_merge_preds)
    parser_merge_preds.set_defaults(handler=_run_merge_preds)

    # Subparser for exporting attributions
    parser_attr = subparsers.add_parser(
        "attr", help="Sequence attribution analysis", formatter_class=_Formatter)
    _general_parser(parser_attr)
    _export_attr_parser(parser_attr)
    parser_attr.set_defaults(handler=_run_attr)

    return parser


def deepdetails(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.function is None:
        parser.print_help()
        parser.exit(1)
    try:
        require_external_binaries()
    except RuntimeError as exc:
        parser.error(str(exc))
    check_update()
    args.handler(args)


if __name__ == "__main__":
    deepdetails()
