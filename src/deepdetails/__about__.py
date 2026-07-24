try:
    from deepdetails._version import __version__
except ImportError:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("DeepDETAILS")
    except PackageNotFoundError:
        __version__ = "0.0.1"
