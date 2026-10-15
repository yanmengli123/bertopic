"""Portable analysis engine; importing this package does not load ML runtimes."""
import importlib.metadata

try:
    # Single source of truth: distribution metadata (wheel/sdist always carry it).
    __version__ = importlib.metadata.version("dsh-bertopic-workflow")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"  # source-tree fallback; doctor reports the divergence


def run_workflow(config_path, **options):
    from .runtime import run
    return run(config_path, **options)
