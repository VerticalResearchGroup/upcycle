import functools
import time
import numpy as np

from .common import *

from . import ops
from . import apps
import logging

logger = logging.getLogger(__name__)
a100_peak_fp16 = 312e12

try:
    import torch
    HAS_TORCH = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import torch.nn.functional as F
    import torch.nn.quantized.functional as qF
    from torch.profiler import profile, record_function, ProfilerActivity

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
        layer = torch.nn.Conv2d(op.c, op.k, op.sf, stride=op.stride, padding=op.pad, bias=False, dtype=torch.float16, device=dev)
    elif op.d == 3:
        layer = torch.nn.Conv3d(op.c, op.k, op.sf, stride=op.stride, padding=op.pad, bias=False, dtype=torch.float16, device=dev)

    x = torch.randn((op.n, op.c, *op.si), dtype=torch.float16, device=dev, requires_grad=False)
    return lambda: layer(x)

# @make_op_func.register
# def _(op : ops.Linear, dev):
#     assert HAS_TORCH
#     assert op.l == 1
#     x = torch.randn((op.m, op.k)).half().to(dev)
#     layer = torch.nn.Linear(op.k, op.n, bias=None).half().to(dev)
#     return lambda: layer(x)

def time_torch_op(upcycle_op, torchdev, niters=100, warmup=10):
    assert HAS_TORCH
    func = make_op_func(upcycle_op, torchdev)
    for _ in range(warmup): func()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        for _ in range(niters):
            func()

        torch.cuda.synchronize()

    avg_time = None
    for x in prof.key_averages():
        if x.key == 'cudaLaunchKernel':
            avg_time = x.cpu_time_total / 1e6 / niters

    assert avg_time is not None
    return avg_time


