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

    x_func = lambda: (torch.randn((op.n, op.c, *op.si), dtype=torch.float16, device=dev, requires_grad=False), )
    return layer, x_func

# @make_op_func.register
# def _(op : ops.Linear, dev):
#     assert HAS_TORCH
#     assert op.l == 1
#     x = torch.randn((op.m, op.k)).half().to(dev)
#     layer = torch.nn.Linear(op.k, op.n, bias=None).half().to(dev)
#     return lambda: layer(x)

def time_torch_op(upcycle_op, dev, niters=100, warmup=10):
    assert HAS_TORCH
    model, x_func = make_op_func(upcycle_op, dev)
    for _ in range(warmup): model(*x_func())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    utils = []

    for _ in range(niters):
        x = x_func()
        torch.cuda.synchronize()
        start.record()
        y = model(*x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
        op_util_a100 = upcycle_op.flops / times[-1] / a100_peak_fp16
        utils.append(op_util_a100)
        print(times[-1], op_util_a100)
        del y

    times = np.array(times)
    utils = np.array(utils)

    avg_time = np.average(times)
    op_util_a100 = upcycle_op.flops / avg_time / a100_peak_fp16
    print(f'{upcycle_op} - {upcycle_op.flops / 1e9:.3f} GOPs:  \t\tavg. time={avg_time:.5f}\tutil={op_util_a100 * 100:.3f}%')
    return times, utils

def run_with_torch(trace : apps.Trace, device_type='cpu', device_id=0, niters=100, warmup=10):
    assert HAS_TORCH
    dev = torch.device(device_type, device_id)

    for op in trace.oplist:
        yield time_torch_op(op, dev, niters=niters, warmup=warmup)



