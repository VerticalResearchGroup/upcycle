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
def make_infer_op_func(op, dev):
    raise NotImplementedError(f'{op} is not supported by make_infer_op_func')

@functools.singledispatch
def make_train_op_func(op, dev):
    raise NotImplementedError(f'{op} is not supported by make_train_op_func')

def make_train_func(layer, *x):
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)

    def train():
        opt.zero_grad()
        loss = layer(*x).sum()
        loss.backward()
        opt.step()

    return train


@make_infer_op_func.register
def _(op : ops.Conv, dev):
    if op.d == 2:
        layer = torch.nn.Conv2d(op.c, op.k, op.sf, stride=op.stride, padding=op.pad, bias=False, dtype=torch.float16, device=dev)
    elif op.d == 3:
        layer = torch.nn.Conv3d(op.c, op.k, op.sf, stride=op.stride, padding=op.pad, bias=False, dtype=torch.float16, device=dev)

    x = torch.randn((op.n, op.c, *op.si), dtype=torch.float16, device=dev)
    return lambda: layer(x)

@make_train_op_func.register
def _(op : ops.Conv, dev):
    if op.d == 2:
        layer = torch.nn.Conv2d(op.c, op.k, op.sf, stride=op.stride, padding=op.pad, bias=False, dtype=torch.float16, device=dev)
    elif op.d == 3:
        layer = torch.nn.Conv3d(op.c, op.k, op.sf, stride=op.stride, padding=op.pad, bias=False, dtype=torch.float16, device=dev)

    x = torch.randn((op.n, op.c, *op.si), dtype=torch.float16, device=dev)
    return make_train_func(layer, x)

@make_infer_op_func.register
def _(op : ops.Linear, dev):
    assert op.l == 1
    layer = torch.nn.Linear(op.k, op.n, bias=None, dtype=torch.float16, device=dev)
    x = torch.randn((op.m, op.k), dtype=torch.float16, device=dev)
    return lambda: layer(x)

@make_train_op_func.register
def _(op : ops.Linear, dev):
    assert op.l == 1
    layer = torch.nn.Linear(op.k, op.n, bias=None, dtype=torch.float16, device=dev)
    x = torch.randn((op.m, op.k), dtype=torch.float16, device=dev)
    return make_train_func(layer, x)

@make_infer_op_func.register
def _(op : ops.Matmul, dev):
    a = torch.randn((op.l, op.m, op.k) if not op.tr_a else (op.l, op.m, op.k), dtype=torch.float16, device=dev)
    b = torch.randn((op.l, op.k, op.n) if not op.tr_b else (op.l, op.n, op.k), dtype=torch.float16, device=dev)

    return lambda: torch.bmm(
        a if not op.tr_a else a.transpose(-1, -2),
        b if not op.tr_b else b.transpose(-1, -2))

@make_train_op_func.register
def _(op : ops.Matmul, dev):
    a = torch.randn((op.l, op.m, op.k) if not op.tr_a else (op.l, op.m, op.k), dtype=torch.float16, device=dev, requires_grad=True)
    b = torch.randn((op.l, op.k, op.n) if not op.tr_b else (op.l, op.n, op.k), dtype=torch.float16, device=dev, requires_grad=True)

    def func():
        c = torch.bmm(
            a if not op.tr_a else a.transpose(-1, -2),
            b if not op.tr_b else b.transpose(-1, -2))

        c.sum().backward()

    return func

@make_infer_op_func.register
def _(op : ops.LstmCell, dev):
    layer = torch.nn.LSTMCell(op.d, op.h, bias=False, dtype=torch.float16, device=dev)
    x = torch.randn((op.n, op.d), dtype=torch.float16, device=dev)
    h = torch.randn((op.n, op.h), dtype=torch.float16, device=dev)
    c = torch.randn((op.n, op.h), dtype=torch.float16, device=dev)
    return lambda: layer(x, (h, c))

@make_train_op_func.register
def _(op : ops.LstmCell, dev):
    layer = torch.nn.LSTMCell(op.d, op.h, bias=False, dtype=torch.float16, device=dev)
    x = torch.randn((op.n, op.d), dtype=torch.float16, device=dev)
    h = torch.randn((op.n, op.h), dtype=torch.float16, device=dev)
    c = torch.randn((op.n, op.h), dtype=torch.float16, device=dev)
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)

    def train():
        opt.zero_grad()
        y, hc = layer(x, (h, c))
        y.sum().backward()
        opt.step()

    return train

def time_torch_op(upcycle_op, torchdev, niters=100, warmup=10):
    assert HAS_TORCH
    if type(upcycle_op) is apps.TrainOp:
        func = make_train_op_func(upcycle_op.fwd_op, torchdev)
    else: func = make_infer_op_func(upcycle_op, torchdev)

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


