"""Inherited by test subprocesses, including lightweight import probes.

Use an audit hook rather than importing torch here: import-contract tests must
be able to prove that importing pubmedqa does not load an ML framework.
"""

import os
import sys

if os.environ.get("PUBMEDQA_OFFLINE_TEST") == "1":

    def deny_network(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError(f"Offline test subprocess forbids {event}")

    sys.addaudithook(deny_network)
