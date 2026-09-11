"""Deterministic control-plane simulator; never initializes a process group.

Each rank executes in its own thread, so mismatched collective order fails with
a bounded timeout instead of silently passing a sequential mock.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, local
from helpers.offline import CpuOnly


class FakeRanks:
    def __init__(self, size=2):
        self.size = size
        self.group = object()
        self.local = local()
        self.barrier = Barrier(size, timeout=5)
        self.values = [None] * size
        self.traces = [[] for _ in range(size)]

    def _exchange(self, kind, value, group):
        if group is not self.group:
            raise AssertionError("Control collective must use the explicit group")
        rank = self.local.rank
        self.traces[rank].append(kind)
        self.values[rank] = deepcopy(value)
        self.barrier.wait()
        result = deepcopy(self.values)
        self.barrier.wait()
        return result

    def broadcast(self, payload, src, group):
        payload[:] = self._exchange("broadcast", payload, group)[src]

    def gather(self, output, value, group):
        output[:] = self._exchange("gather", value, group)

    def run(self, operation):
        def worker(rank):
            self.local.rank = rank
            try:
                with CpuOnly():
                    return operation(rank)
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=self.size) as pool:
            return list(pool.map(worker, range(self.size)))
