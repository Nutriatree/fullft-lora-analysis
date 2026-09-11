"""Run the unittest suite with network and accelerator execution blocked."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
import tempfile
import atexit

ROOT = Path(__file__).resolve().parents[1]
_cache = tempfile.TemporaryDirectory(prefix="pubmedqa-test-cache-")
atexit.register(_cache.cleanup)
os.environ.update(
    HF_HOME=_cache.name,
    HF_DATASETS_CACHE=str(Path(_cache.name) / "datasets"),
    MPLCONFIGDIR=str(Path(_cache.name) / "matplotlib"),
)
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
    CUDA_VISIBLE_DEVICES="",
)
os.environ["PUBMEDQA_OFFLINE_TEST"] = "1"
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT / "tests/helpers"), str(ROOT / "src"), str(ROOT)]
)
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from helpers.offline import offline_cpu

# Existing import tests sometimes set PYTHONPATH=src explicitly. Inject the
# guard path when starting any Python subprocess so it cannot escape the gate.
import subprocess
from unittest.mock import patch

_popen = subprocess.Popen


def guarded_popen(*args, **kwargs):
    env = dict(kwargs.get("env") or os.environ)
    env["PUBMEDQA_OFFLINE_TEST"] = "1"
    env["PYTHONPATH"] = (
        str(ROOT / "tests/helpers") + os.pathsep + env.get("PYTHONPATH", "")
    )
    kwargs["env"] = env
    return _popen(*args, **kwargs)


def main() -> None:
    with offline_cpu(), patch("subprocess.Popen", guarded_popen):
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    # Skips must not silently turn missing CPU/PEFT coverage into a passing gate.
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)


if __name__ == "__main__":
    main()
