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

    import torch.nn.functional as F
    import torch.nn.quantized.functional as qF

except ImportError:
    logger.warn(f'Torch not found, run_with_torch will be unavailable.')
    HAS_TORCH = False

@functools.singledispatch
def make_op_func(op, dev):
    raise NotImplementedError(f'{op} is not supported by make_torch_op_input')

@make_op_func.register
def _(op : ops.Conv, dev):
    assert HAS_TORCH
    if op.d == 2:
        layer = torch.nn.Conv2d(op.c, op.k, op.sf, stride=op.stride, padding=op.pad, bias=None).half().to(dev)
    elif op.d == 3:
        layer = torch.nn.Conv3d(op.c, op.k, op.sf, stride=op.stride, padding=op.pad, bias=None).half().to(dev)

    x = torch.randn((op.n, op.c, *op.si)).half().to(dev)
    y = layer(x)

    assert y.shape == (op.n, op.k, *op.so)
    return lambda: layer(x)

@make_op_func.register
def _(op : ops.Linear, dev):
    assert HAS_TORCH
    assert op.l == 1
    x = torch.randn((op.m, op.k)).half().to(dev)
    layer = torch.nn.Linear(op.k, op.n, bias=None).half().to(dev)
    return lambda: layer(x)

def time_torch_op(upcycle_op, dev, niters=100):
    assert HAS_TORCH
    f = make_op_func(upcycle_op, dev)
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(niters):
        f()
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / niters

def run_with_torch(trace : apps.Trace, device_type='cpu', device_id=0, niters=100):
    assert HAS_TORCH
    dev = torch.device(device_type, device_id)

    return np.array([
        time_torch_op(op, dev, niters=niters)
        for op in trace.oplist
    ])


