import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherCommandsTest(unittest.TestCase):
    def test_single_ddp_fsdp_and_smoke_command_generation_without_training(self):
        for mode, script in (
            ("single", "run_pubmedqa_full_study.sh"),
            ("ddp", "run_pubmedqa_full_study.sh"),
            ("fsdp", "run_pubmedqa_full_study.sh"),
            ("fsdp", "run_pubmedqa_fsdp_smoke.sh"),
        ):
            with (
                self.subTest(mode=mode, script=script),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                stub = (ROOT / "tests/helpers/launcher_stub.py").read_text()
                for name in ("python", "torchrun"):
                    entry = root / name
                    entry.write_text(f"#!{sys.executable}\n" + stub)
                    entry.chmod(0o700)
                data = root / "synthetic.jsonl"
                data.write_text("{}\n")
                command_log = root / "commands.jsonl"
                env = {
                    **os.environ,
                    "PATH": str(root) + os.pathsep + os.environ["PATH"],
                    "PUBMEDQA_COMMAND_LOG": str(command_log),
                    "PUBMEDQA_RUN_ID": "commands",
                    "PUBMEDQA_DISTRIBUTED_MODE": mode,
                    "PUBMEDQA_GPU_IDS": "0,1",
                    "PUBMEDQA_TRAIN_FILE": str(data),
                    "PUBMEDQA_VALIDATION_FILE": str(data),
                    "PUBMEDQA_TEST_FILE": str(data),
                    "PUBMEDQA_TRAIN_OUTPUT_DIR": str(root / "train"),
                    "PUBMEDQA_BASELINE_OUTPUT_DIR": str(root / "eval"),
                    "PUBMEDQA_INCLUDE_LL2": "1",
                    "PUBMEDQA_INCLUDE_LOW_DATA": "0",
                }
                result = subprocess.run(
                    ["bash", str(ROOT / "scripts" / script)],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                commands = [
                    json.loads(line) for line in command_log.read_text().splitlines()
                ]
                training = [
                    args
                    for args in commands
                    if "scripts/run_pubmedqa_experiments.py" in args
                ]
                self.assertGreater(len(training), 0)
                for args in training:
                    self.assertEqual(
                        "python" if mode == "single" else "torchrun", args[0]
                    )
                    self.assertEqual(mode, args[args.index("--distributed-mode") + 1])
                    if mode != "single":
                        self.assertIn("--nproc_per_node=2", args)
