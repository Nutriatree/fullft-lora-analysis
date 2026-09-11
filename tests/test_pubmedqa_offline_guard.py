import socket
import unittest
import torch

from helpers.offline import offline_cpu


class OfflineGuardTest(unittest.TestCase):
    def test_network_and_device_allocations_fail_closed(self):
        with offline_cpu():
            with self.assertRaisesRegex(AssertionError, "forbids"):
                socket.create_connection(("example.invalid", 443))
            for device in ("cuda", "mps"):
                with self.assertRaisesRegex(AssertionError, "forbids"):
                    torch.ones(1, device=device)
            self.assertEqual(1.0, torch.ones(1).item())
