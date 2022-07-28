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
def _(conv : ops.Conv2D, dev):
    assert HAS_TORCH
    x = torch.randn((conv.n, conv.c, conv.h, conv.w), device=dev)
    f = torch.randn((conv.k, conv.c, conv.r, conv.s), device=dev)
    y = F.conv2d(x, f, padding=conv.pad, stride=conv.stride)

    assert y.shape == (conv.n, conv.k, conv.p, conv.q)

    if conv.dtype == Dtype.I8:
        raise NotImplementedError()

    elif conv.dtype == Dtype.FP16:
        fp16_x = x.to(torch.float16).to(dev)
        fp16_f = f.to(torch.float16).to(dev)

        func = lambda: F.conv2d(fp16_x, fp16_f, padding=conv.pad, stride=conv.stride, bias=None)

    return func

@make_op_func.register
def _(lin : ops.Linear, dev):
    assert HAS_TORCH
    x = torch.randn((lin.n, lin.c), device=dev)
    y = torch.randn((lin.n, lin.k), device=dev)

    if lin.dtype == Dtype.I8:
        raise NotImplementedError()

    elif lin.dtype == Dtype.FP16:
        fp16_x = x.to(torch.float16).to(dev)
        fp16_f = f.to(torch.float16).to(dev)

        func = lambda: F.conv2d(fp16_x, fp16_f, padding=conv.pad, stride=conv.stride, bias=None)

    return func

def time_torch_op(upcycle_op, dev, niters=1):
    assert HAS_TORCH
    f = make_op_func(upcycle_op, dev)
    t0 = time.perf_counter()
    for _ in range(niters): f()
    t1 = time.perf_counter()
    return (t1 - t0) / niters


def run_with_torch(trace : apps.Trace, device_type='cpu', device_id=0, niters=100):
    assert HAS_TORCH
    dev = torch.device(device_type, device_id)

    return np.array([
        time_torch_op(op, dev, niters=niters)
        for op in trace.oplist
    ])


