"""Argument recorder copied as python/torchrun by launcher command tests.

It never evaluates -c, imports training code or starts a subprocess. This is a
shell command-generation test, not a simulated successful GPU run.
"""
import json
import os
from pathlib import Path
import sys

with Path(os.environ["PUBMEDQA_COMMAND_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + "\n")
if len(sys.argv) > 2 and sys.argv[1] == "-c" and "print(" in sys.argv[2]:
    print("0")  # synthetic selective-layer selection for the next command
