"""argparse ``type=`` helpers for validating/converting CLI arguments.

This module intentionally has no dependencies beyond the standard library.
It is imported eagerly by ``deepdetails.cli`` to build the argument parser
(including for ``--help``), so pulling in heavy packages here (torch,
pytorch_lightning, wandb, pybedtools, ...) would slow down every CLI
invocation.
"""
import argparse
import os
from typing import Sequence


def existing_file(path: str) -> str:
    """argparse ``type=`` helper: require ``path`` to point to an existing file

    Parameters
    ----------
    path : str
        Value of the option

    Returns
    -------
    path : str
        The unmodified value, if it points to an existing file

    Raises
    ------
    argparse.ArgumentTypeError
        If ``path`` does not point to an existing file. Unlike a plain
        ``OSError``, this is caught by argparse and surfaced as a normal
        ``usage: ... error: ...`` message instead of an unhandled traceback.
    """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"The file {path} does not exist")
    return path


def existing_dir(path: str) -> str:
    """argparse ``type=`` helper: require ``path`` to point to an existing directory

    Parameters
    ----------
    path : str
        Value of the option

    Returns
    -------
    path : str
        The unmodified value, if it points to an existing directory

    Raises
    ------
    argparse.ArgumentTypeError
        If ``path`` does not point to an existing directory.
    """
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"The directory {path} does not exist")
    return path


def created_dir(path: str) -> str:
    """argparse ``type=`` helper: ensure ``path`` is a directory, creating it if needed

    Parameters
    ----------
    path : str
        Value of the option

    Returns
    -------
    path : str
        The unmodified value, after the directory has been created if it
        did not already exist

    Raises
    ------
    argparse.ArgumentTypeError
        If ``path`` does not exist and cannot be created.
    """
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            raise argparse.ArgumentTypeError(
                f"The directory {path} does not exist and cannot be created"
            ) from exc
    return path


def alpha_as_fraction(value: str) -> float:
    """argparse ``type=`` helper for ``--alpha``/``--redundancy-loss-coef``

    Users supply this on a 0-100 scale; internally it is used directly as the
    redundancy loss coefficient, which matches our experiment scale of 0-1, so
    the parsed value is divided by 100 here.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``value`` is not a number in [0, 100].
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--alpha/--redundancy-loss-coef must be a number between 0 and 100, got {value!r}"
        ) from exc
    if not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError(
            f"--alpha/--redundancy-loss-coef must be between 0 and 100, got {value}"
        )
    return parsed / 100

class DevicesAction(argparse.Action):
    def __call__(
            self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
            values: Sequence[str], option_string: str | None = None):
        if values == ["auto"]:
            setattr(namespace, self.dest, "auto")
            return

        devices = []
        for value in values:
            try:
                devices.append(int(value))
            except ValueError as exc:
                raise argparse.ArgumentError(
                    self, "devices must be integers or the single value 'auto'"
                ) from exc
        setattr(namespace, self.dest, devices)
