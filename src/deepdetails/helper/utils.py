import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import unicodedata
from datetime import datetime
from typing import List, Optional, Sequence, Union
from urllib.request import urlopen

import numpy as np
import pybedtools
import pytorch_lightning as pl
import torch
import wandb
from lightning_fabric.loggers import Logger as FabricLogger
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.loggers import Logger as PLLogger

from deepdetails.par_description import PARAM_DESC

LightningLogger = PLLogger | FabricLogger

REQUIRED_BINARIES = ("awk", "bedtools", "bedGraphToBigWig", "sort")
_PYPI_JSON_URL = "https://pypi.org/pypi/DeepDETAILS/json"
_CONDA_UPDATE_CMD = "conda update -c bioconda -c conda-forge deepdetails"
_PIP_UPDATE_CMD = "pip install -U DeepDETAILS"

logger = logging.getLogger(__name__)


def require_external_binaries() -> None:
    """Raise ``RuntimeError`` if any required external tool is missing from ``PATH``."""
    missing = [tool for tool in REQUIRED_BINARIES if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Required external tool(s) not found on PATH: "
            f"{', '.join(missing)}. Install them and ensure they are callable "
            "before re-running."
        )


def check_update(timeout: float = 5.0) -> None:
    """Compare the installed version against the latest PyPI release.

    Network failures are logged and ignored so offline / CI use is unaffected.
    Set ``DEEPDETAILS_SKIP_UPDATE_CHECK=1`` to skip this check.
    """
    if os.environ.get("DEEPDETAILS_SKIP_UPDATE_CHECK", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return

    try:
        from deepdetails.__about__ import __version__ as local_version

        with urlopen(_PYPI_JSON_URL, timeout=timeout) as response:
            remote_version = (
                json.loads(response.read().decode()).get("info", {}).get("version")
            )
        if not remote_version:
            return

        local_base = local_version.split("+", 1)[0]
        if local_base == remote_version:
            logger.info("You are using the latest DeepDETAILS release (%s)", local_base)
            return

        try:
            from packaging.version import Version

            outdated = Version(local_base) < Version(remote_version)
        except Exception:
            outdated = True

        if outdated:
            logger.warning(
                "Your DeepDETAILS version is out of date (%s vs. %s). "
                "Update with conda (recommended): `%s`. "
                "Or with pip: `%s`.",
                local_base,
                remote_version,
                _CONDA_UPDATE_CMD,
                _PIP_UPDATE_CMD,
            )
        else:
            logger.info(
                "Installed DeepDETAILS (%s) is newer than the latest PyPI release (%s)",
                local_base,
                remote_version,
            )
    except Exception as exc:
        logger.debug("Skipping update check: %s", exc)


def run_command(cmd: Union[str, Sequence[str]], raise_exception: bool = False):
    """Run command

    Parameters
    ----------
    cmd : Union[str, Sequence[str]]

    raise_exception : bool
        Raise an exception if the return code is not 0.

    Returns
    -------
    stdout : str

    stderr : str

    return_code : int

    """
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    proc = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if raise_exception and proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout, proc.stderr, proc.returncode


def _pipeline_stage_failed(returncode: int | None, last: bool) -> bool:
    if returncode == 0:
        return False
    if returncode is None:
        return True
    # Upstream SIGPIPE is expected when a later stage exits first.
    if not last and returncode == -signal.SIGPIPE:
        return False
    return True


def run_pipeline(commands: Sequence[Sequence[str]], stdout) -> None:
    """Run ``commands[0] | commands[1] | ...`` into ``stdout`` without a shell.

    Each stage's exit code is checked. Stderr is captured to temp files so a
    noisy tool cannot fill a PIPE and deadlock the pipeline.

    Parameters
    ----------
    commands : Sequence[Sequence[str]]
        List of commands to run
    stdout : file-like object
        Output file

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If any command fails.
    """
    err_files = []
    procs: list[subprocess.Popen] = []
    try:
        prev_out = None
        for i, cmd in enumerate(commands):
            err = tempfile.TemporaryFile()
            err_files.append(err)
            last = i == len(commands) - 1
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL if prev_out is None else prev_out,
                stdout=stdout if last else subprocess.PIPE,
                stderr=err,
            )
            if prev_out is not None:
                prev_out.close()
            procs.append(proc)
            prev_out = None if last else proc.stdout

        for proc in reversed(procs):
            proc.wait()

        for i, (cmd, proc, err) in enumerate(zip(commands, procs, err_files)):
            last = i == len(commands) - 1
            if not _pipeline_stage_failed(proc.returncode, last):
                continue
            err.seek(0)
            msg = err.read().decode("utf-8", errors="replace").strip()
            detail = f": {msg}" if msg else ""
            raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}){detail}")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for err in err_files:
            err.close()


def get_trainer(
    study_name: str,
    save_to: str = ".",
    min_delta: float = 0,
    earlystop_patience: int = 3,
    max_epochs: int = 200,
    save_top_k_model: Union[str, int] = 1,
    hide_progress_bar: bool = False,
    model_summary_depth: int = 1,
    version: Optional[str] = None,
    accelerator: str = "auto",
    devices: Union[Sequence[int], str, int] = "auto",
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_upload_model: Union[str, bool] = False,
    wandb_online: bool = False,
    pass_mark: str = "1st",
    training_readout: str = "train_loss",
) -> tuple[pl.Trainer, str]:
    """
    Get pl.Trainer for training / inference, etc.

    Parameters
    ----------
    study_name : str
        {study_name}
    save_to : str
        {save_to}
    min_delta : float
        {min_delta}
    earlystop_patience : int
        {earlystop_patience}
    max_epochs : int
        {max_epochs}
    save_top_k_model : Union[str, int]
        {save_top_k_model}
    hide_progress_bar : bool
        {hide_progress_bar}
    model_summary_depth : int
        {max_depth}
    version : Optional[str]
        {wandb_version}
    accelerator : Optional[str]
        {accelerator}
    devices : Union[List[int], str, int]
        {devices}
    wandb_project : Optional[str]
        {wandb_project}
    wandb_entity : Optional[str]
        {wandb_entity}
    wandb_upload_model : Union[str, int]
        {wandb_upload_model}
    wandb_online : bool
        {wandb_online}
    pass_mark
    training_readout : str
        Metric monitored for checkpointing and early stopping.

    Returns
    -------
    trainer_obj : pl.Trainer
        Lightning Trainer object
    wbl.version : str
        Final effective WandB version string
    """.format(**PARAM_DESC)
    pass_str = f"_{pass_mark}" if pass_mark else ""
    ver_str = (
        f"{version}{pass_str}" if version else datetime.now().strftime("%y%m%d%H%M%S")
    )

    # Normalize sequences to list for Lightning's Trainer typing.
    trainer_devices: Union[list[int], str, int]
    if isinstance(devices, (str, int)):
        trainer_devices = devices
    else:
        trainer_devices = list(devices)

    # CPUAccelerator expects `devices` to be a plain positive int, not a list.
    is_cpu_accelerator = accelerator == "cpu" or (
        accelerator == "auto" and not torch.cuda.is_available()
    )
    if is_cpu_accelerator and isinstance(trainer_devices, list):
        trainer_devices = max(len(trainer_devices), 1)

    is_multi_gpu = (
        isinstance(trainer_devices, list)
        and len(trainer_devices) > 1
        and accelerator != "cpu"
    )

    if not is_multi_gpu:
        # Close any leftover run so retries do not reuse the previous record.
        wandb.finish()
        wbl = WandbLogger(
            name=f"{study_name}{pass_str}",
            project=wandb_project,
            version=ver_str,
            reinit=True,
            entity=wandb_entity,
            log_model=wandb_upload_model,  # pyrefly: ignore[bad-argument-type]
            save_dir=save_to,
            offline=not wandb_online,
        )
        ver = str(wbl.version)
    else:
        wbl = None
        ver = ver_str

    csvl = CSVLogger(
        name=f"{study_name}{pass_str}",
        version=ver,
        save_dir=save_to,
        prefix=f"{study_name}{pass_str}",
    )

    save_top_k = (
        save_top_k_model if isinstance(save_top_k_model, int) else int(save_top_k_model)
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=training_readout,
        save_top_k=save_top_k,
        dirpath=os.path.join(save_to, study_name, ver),
    )
    early_stop_callback = EarlyStopping(
        monitor=training_readout,
        min_delta=min_delta,
        patience=earlystop_patience,
        verbose=True,
        mode="min",
    )
    callbacks: list[Callback] = [
        early_stop_callback,
        checkpoint_callback,
        ModelSummary(max_depth=model_summary_depth),
    ]
    trainer_obj = pl.Trainer(
        logger=[wbl, csvl] if wbl is not None else [csvl],
        enable_checkpointing=True,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=trainer_devices,
        callbacks=callbacks,
        enable_progress_bar=False if hide_progress_bar else True,
    )
    return trainer_obj, ver


def internal_qc(
    metrics: list[float], pred_counts: torch.Tensor
) -> tuple[tuple[float, list[float]], bool, bool]:
    """
    Run internal QC to determine if the deconvolution is sound

    Parameters
    ----------
    metrics : list[float]
        List of internal metric values
    pred_counts : torch.Tensor
        Accumulated predicted counts for each target in a strand-specific manner

    Returns
    -------
    qc_payload : tuple[float, list[float]]
        ``(qc_val, predicted counts as floats)``. ``qc_val`` is 0.0 when fewer
        than three metrics were collected.
    branch_corr_qc_passed : bool
        Branch-correlation verdict. True when the metric decreased, late
        correlation is already low, or there were fewer than three
        samples (a warning is logged in that last case).
    sum_qc_passed : bool
        True when every cluster/strand has non-zero predicted mass.
    """
    qc_val = 0.0
    later = 0.0
    if len(metrics) > 20:
        obs = metrics[:10] + metrics[-10:]
    elif len(metrics) > 2:
        obs = metrics
    else:
        obs = None
    if obs is not None:
        n_steps = len(obs)
        early = np.mean(obs[: n_steps // 2]) + 10e-16
        later = np.mean(obs[n_steps // 2 :]) + 10e-16
        qc_val = float(early / later)
        branch_corr_qc_passed = bool(qc_val > 1.0 or float(later) < 0.4)
    else:
        logger.warning(
            "Correlation-based QC skipped: fewer than 3 metric values collected."
        )
        branch_corr_qc_passed = True

    sum_qc_passed = (
        torch.isclose(pred_counts, torch.zeros_like(pred_counts), atol=0.1).sum().item()
        == 0
    )
    pred_as_floats = [float(v) for v in pred_counts.flatten().tolist()]
    return (qc_val, pred_as_floats), branch_corr_qc_passed, sum_qc_passed


def calc_counts_per_locus(
    profiles: Union[tuple[torch.Tensor, ...], List[torch.Tensor]],
    counts: Union[tuple[torch.Tensor, ...], List[torch.Tensor]],
    is_per_cluster_profile: bool = False,
) -> torch.Tensor:
    """Calculate read counts per genomic locus

    Parameters
    ----------
    profiles : Union[tuple[torch.Tensor, ...], List[torch.Tensor]]
        List of `torch.Tensor`, each Tensor stores the unnormed predictions for a cluster
    counts : Union[tuple[torch.Tensor, ...], List[torch.Tensor]]
        List of `torch.Tensor`, each Tensor stores the read counts for a cluster
    is_per_cluster_profile : bool, optional
        Set this as True if you want the function to return cluster-specific predictions, by default False

    Returns
    -------
    torch.Tensor
        Transformed predictions. Shape: clusters, batch, strands, seq_len if is_per_cluster_profile is True
        batch, strands, seq_len if is_per_cluster_profile if False
    """
    stacked_profiles = torch.stack(profiles, dim=0)
    stacked_counts = torch.stack(counts, dim=0)

    reshaped_counts = stacked_counts.repeat(1, 1, stacked_profiles.shape[-1]).view(
        stacked_counts.shape[0],
        stacked_counts.shape[1],
        -1,
        stacked_counts.shape[2],
    )
    reshaped_counts = torch.swapaxes(reshaped_counts, 2, 3)

    # cluster-specific predictions
    preds = stacked_profiles * reshaped_counts

    if not is_per_cluster_profile:
        # aggregated predictions
        preds = preds.sum(dim=0)
    return preds


def rescaling_prediction(
    pc_profiles: list[torch.Tensor],
    pc_counts: list[torch.Tensor],
    expected_bulk_counts: torch.Tensor,
    expected_bulk_profiles: torch.Tensor,
    rescaling_mode: int = 0,
) -> np.ndarray:
    """Rescale predictions based on the observed bulk profiles

    Parameters
    ----------
    pc_profiles : list[torch.Tensor]
        List of predicted profiles. len(pc_profiles): clusters.
        Shape of elements in the list: batch, strands, seq_len
    pc_counts : list[torch.Tensor]
        List of predicted counts. len(pc_profiles): clusters.
        Shape of elements in the list: batch, strands
    expected_bulk_counts : torch.Tensor
        Observed bulk counts. Shape: batch, strands
    expected_bulk_profiles : torch.Tensor
        Observed bulk profiles. Shape: batch, strands, seq_len
    rescaling_mode : int
        0: No rescaling
        1: Rescaled by bulk counts
        2: Rescaled by bulk profiles

    Returns
    -------
    cluster_preds: torch.Tensor
        Profile prediction for each cluster. Shape: clusters, batch, strands, seq_len
    """
    cluster_preds = calc_counts_per_locus(pc_profiles, pc_counts, True).cpu().numpy()

    if rescaling_mode == 1:  # total counts
        # predicted bulk counts
        ps_counts = torch.stack(pc_counts).sum(dim=0)
        # calculate rescale factor (k)
        # k * y_hat = y
        k = expected_bulk_counts / torch.clamp(ps_counts, min=1e-15)
        # get the rescaled counts
        pc_counts = [pcc * k for pcc in pc_counts]
        cluster_preds = (
            calc_counts_per_locus(pc_profiles, pc_counts, True).cpu().numpy()
        )
    elif rescaling_mode == 2:  # per-bp counts
        bulk_preds = cluster_preds.sum(axis=0)
        # calculate rescale factor (k)
        # k * y_hat = y
        k = expected_bulk_profiles.cpu() / (bulk_preds + 1e-15)
        # rescale
        cluster_preds = (k * cluster_preds).numpy()
    return cluster_preds


def get_log_dir(logger: LightningLogger | None):
    """
    Get log directory

    Parameters
    ----------
    logger : pytorch_lightning.loggers.Logger
        The logger instance

    Returns
    -------
    log_dir : str
        Log directory
    version : str
        Model/logger version
    """
    try:
        if isinstance(logger, loggers.WandbLogger):
            log_dir = os.path.join(
                logger.save_dir or ".",
                logger._name or "",
                str(logger.version or ""),
            )
            version = logger.version
        elif isinstance(logger, loggers.TensorBoardLogger):
            log_dir = logger.log_dir
            version = str(logger.version)
        else:
            log_dir = "."
            version = ""
    except Exception:
        log_dir = "."
        version = ""
    return log_dir, version


def transform_counts(
    values: torch.Tensor, inject_random_noise: float = 10e-16, method: str = "asinh"
) -> torch.Tensor:
    """
    Get asinh/log- transformed counts

    Parameters
    ----------
    values : torch.Tensor
        Values to be transformed
    inject_random_noise : float
        Scale of the random noise to be injected. If you just want to do asinh transformation, set this as 0.
    method : str
        Transformation method. Can be 'asinh' or 'log1p'

    Returns
    -------
    transformed_values: torch.Tensor
        The same shape as `values`
    """
    noises = torch.randn_like(values, device=values.device) * inject_random_noise
    func = torch.asinh if method == "asinh" else torch.log1p
    return func(values) + noises


def slugify(value: str, allow_unicode: bool = False) -> str:
    """Converts a string to a safe path string by:

        1. Converting to ASCII if `allow_unicode` is False (the default).
        2. Converting to lowercase.
        3. Removing characters that aren’t alphanumerics, underscores, hyphens, or whitespace.
        4. Replacing any whitespace or repeated dashes with single dashes.
        5. Removing leading and trailing whitespace, dashes, and underscores.
    Parameters
    ----------
    value : str
        String to be converted
    allow_unicode : bool
        Convert to ASCII if `allow_unicode` is False

    References
    ----------
    - https://github.com/django/django/blob/5f180216409d75290478c71ddb0ff8a68c91dc16/django/utils/text.py#L452-L469

    Returns
    -------
    slugified_str : str
        Converted string
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def set_tmp_for_pbt(tmp_dir="."):
    current_tmp_dir = pybedtools.get_tempdir()
    # if pybedtools is using the system's default tmp dir.,
    # then switch to current working directory to avoid using up all spaces at `/`
    if current_tmp_dir in (
        "/tmp",
        "/var/tmp",
        "/usr/tmp",
        "C:\\TEMP",
        "C:\\TMP",
        "\\TMP",
    ):
        pybedtools.set_tempdir(tmp_dir)


def _is_unsorted_bedgraph_error(message: str) -> bool:
    """
    True if bedGraphToBigWig rejected the file for sort order.

    Parameters
    ----------
    message : str
        The error message from bedGraphToBigWig

    Returns
    -------
    bool
    """

    _UNSORTED_BEDGRAPH_MARKERS = (
        "not sorted",
        "not case-sensitive sorted",
        "not in single block",
    )

    return any(marker in message for marker in _UNSORTED_BEDGRAPH_MARKERS)


def bedgraph_to_bigwig(
    in_bedgraph_path: str, out_bigwig_path: str, chrom_size_path: str
):
    """
    Convert a file in bedGraph format to bigWig format

    Parameters
    ----------
    in_bedgraph_path : str

    out_bigwig_path : str

    chrom_size_path : str


    Returns
    -------

    """
    # get chromosomes that have size info
    allowed_chromosomes = set()
    with open(chrom_size_path, "r") as csf:
        for line in csf:
            allowed_chromosomes.add(line.strip().split()[0])

    tmp_file = f"{in_bedgraph_path}.1"
    sorted_file = f"{tmp_file}.sorted"
    try:
        with (
            open(in_bedgraph_path, "r") as input_file,
            open(tmp_file, "w") as output_file,
        ):
            for line in input_file:
                parts = line.split("\t")
                chromosome = parts[0]
                if chromosome in allowed_chromosomes:
                    output_file.write(line)

        try:
            run_command(
                ["bedGraphToBigWig", tmp_file, chrom_size_path, out_bigwig_path],
                raise_exception=True,
            )
        except RuntimeError as e:
            if not _is_unsorted_bedgraph_error(str(e)):
                raise
            try:
                # fmt: off
                run_command(
                    ["env", "LC_ALL=C", "sort", "-o", sorted_file, "-k1,1", "-k2,2n", tmp_file],
                    raise_exception=True,
                )
                run_command(
                    ["bedGraphToBigWig", sorted_file, chrom_size_path, out_bigwig_path],
                    raise_exception=True,
                )
                # fmt: on
            except RuntimeError as retry_err:
                raise retry_err from e
    finally:
        for path in (tmp_file, sorted_file):
            if os.path.exists(path):
                os.remove(path)


def compare_dicts(dicts: Sequence[dict]) -> bool:
    """
    Compare a list of dictionaries where the values may be numpy arrays.
    Returns True if all dictionaries have identical keys and values,
    False otherwise.

    Parameters
    ----------
    dicts : Sequence[dict]

    """
    if not dicts:
        return True  # empty list of dictionaries is trivially "identical"

    # get the keys of the first dictionary to compare against
    keys = list(dicts[0].keys())

    # compare keys across all dictionaries
    for d in dicts:
        if set(d.keys()) != set(keys):
            return False

    # compare values for each key across all dictionaries
    for key in keys:
        # get the value from the first dictionary
        first_value = dicts[0][key]

        # check if the value for this key in all dictionaries is the same
        for d in dicts[1:]:
            value = d[key]
            # if both values are numpy arrays, use np.array_equal for comparison
            if isinstance(first_value, np.ndarray) and isinstance(value, np.ndarray):
                if not np.array_equal(first_value, value):
                    return False
            # otherwise, perform regular equality check
            elif first_value != value:
                return False

    return True
