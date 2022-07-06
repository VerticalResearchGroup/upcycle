import functools
import time
import numpy as np

from .common import *

from . import ops
from . import apps
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True

except ImportError:
    logger.warn(f'Torch not found, run_with_torch will be unavailable.')
    HAS_TORCH = False

@functools.singledispatch
def make_torch_op(op, dev):
    raise NotImplementedError(f'{op} is not supported by make_torch_op')

@make_torch_op.register
def _(conv : ops.Conv2D, dev):
    assert HAS_TORCH
    return torch.nn.Conv2d(conv.c, conv.k, conv.r, conv.stride, conv.pad, device=dev)

@functools.singledispatch
def make_torch_op_input(op, dev):
    raise NotImplementedError(f'{op} is not supported by make_torch_op_input')

@make_torch_op_input.register
def _(conv : ops.Conv2D, dev):
    assert HAS_TORCH
    return torch.randn((conv.n, conv.c, conv.h, conv.w), device=dev)

def time_torch_op(upcycle_op, dev, niters=1):
    assert HAS_TORCH
    op = make_torch_op(upcycle_op, dev)
    x = make_torch_op_input(upcycle_op, dev)
    t0 = time.perf_counter()
    for _ in range(niters): op(x)
    t1 = time.perf_counter()
    return (t1 - t0) / niters


def run_with_torch(trace : apps.Trace, device_type='cpu', niters=100):
    assert HAS_TORCH
    dev = torch.device(device_type)

    return np.array([
        time_torch_op(op, dev, niters=niters)
        for op in trace.oplist
    ])


