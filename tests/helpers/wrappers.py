"""CPU wrappers emulate API dispatch only, not distributed numerical reductions."""

from contextlib import nullcontext
import torch


class CPUWrapper(torch.nn.Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module
        self.options = kwargs
        self.collections = 0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def no_sync(self):
        return nullcontext()

    def clip_grad_norm_(self, max_norm):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def state_dict(self, *args, **kwargs):
        self.collections += 1
        return self.module.state_dict(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.module.save_pretrained(*args, **kwargs)

    @staticmethod
    def state_dict_type(*args, **kwargs):
        return nullcontext()

    @staticmethod
    def summon_full_params(*args, **kwargs):
        return nullcontext()
