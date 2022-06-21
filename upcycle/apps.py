from dataclasses import dataclass

from .common import *
from . import ops

@dataclass
class Trace:
    oplist : list[ops.Operator]

    @property
    def flops(self): return sum(op.flops for op in self.oplist)

    def infer(self):
        for i, op in enumerate(self.oplist):
            if type(op) is ops.Linear:
                self.oplist[i] = ops.Linear(
                    op.dtype, op.train, op.l, op.m, op.n, op.k, False, True)

            elif type(op) is ops.Conv2D:
                self.oplist[i] = ops.Conv2D(
                    op.dtype, op.train,
                    op.n, op.h, op.w, op.c, op.p, op.q, op.k, op.r, op.s,
                    op.stride, op.pad, False)

            elif type(op) is ops.Lstm:
                self.oplist[i] = ops.Lstm(
                    op.dtype, op.train, op.n, op.s, op.d, op.h, False, True)

        return self

    def train(self, bwd_only=False):
        for i, op in enumerate(self.oplist):
            if type(op) is ops.Linear:
                self.oplist[i] = ops.Linear(
                    op.dtype, op.train, op.l, op.m, op.n, op.k, False, False)

            elif type(op) is ops.Conv2D:
                self.oplist[i] = ops.Conv2D(
                    op.dtype, op.train,
                    op.n, op.h, op.w, op.c, op.p, op.q, op.k, op.r, op.s,
                    op.stride, op.pad, True)

            elif type(op) is ops.Lstm:
                self.oplist[i] = ops.Lstm(
                    op.dtype, op.train, op.n, op.s, op.d, op.h, False, False)

        bwd_list = []
        for op in reversed(self.oplist):
            if op.train:
                if type(op) in ops.backward_map:
                    backward_ops = ops.backward_map[type(op)]
                    for bop in backward_ops:
                        bwd_list.append(bop.from_forward(op))

                else:
                    logging.warn(f'No backward operator for {type(op)}. Ignoring.')

        if bwd_only: self.oplist = bwd_list
        else: self.oplist = self.oplist + bwd_list

        return self

    @property
    def unique_ops(self):
        return set(self.oplist)

def testmatmul(dtype, n=1):
    return Trace([ ops.Linear(dtype, True, 1, n, 1024, 1024, False, False) ])

def testconv_fwd(dtype, n=1):
    return Trace([ ops.Conv2D(dtype, True, n, 224, 224, 3, 112, 112, 64, 7, 7, 2, 3, False) ])

def testconv_bwd(dtype, n=1):
    return Trace([ ops.Conv2DBwd(dtype, True, n, 224, 224, 3, 112, 112, 64, 7, 7, 2, 3, False) ])

# MLPerf Apps based on v2.0 Inference and v1.1 Training datacenter benchmarks:
# https://mlcommons.org/en/inference-datacenter-20/


def bertlarge(dtype, n=1, s=512):
    return Trace([
        ops.Linear(dtype, True, 1, n * s, 1024, 1024, False, False),
        ops.Linear(dtype, True, 1, n * s, 1024, 1024, False, False),
        ops.Linear(dtype, True, 1, n * s, 1024, 1024, False, False),
        ops.Matmul(dtype, True, n * 16, s, s, 64, False, True),
        ops.Matmul(dtype, True, n * 16, s, 64, s, False, True),
        ops.Linear(dtype, True, 1, n * s, 1024, 1024, False, False),
        ops.Linear(dtype, True, 1, n * s, 4096, 1024, False, False),
        ops.Linear(dtype, True, 1, n * s, 1024, 4096, False, False),
    ] * 24)

def bertlarge384(dtype, n=1, s=384): return bertlarge(dtype, n, s)

def bertlarge_squadavg(dtype, n=1, s=178): return bertlarge(dtype, n, s)

def bertbase(dtype, n=1, s=512):
    return Trace([
        ops.Linear(dtype, True, 1, n * s, 768, 768, False, False),
        ops.Linear(dtype, True, 1, n * s, 768, 768, False, False),
        ops.Linear(dtype, True, 1, n * s, 768, 768, False, False),
        ops.Matmul(dtype, True, n * 12, s, s, 64, False, True),
        ops.Matmul(dtype, True, n * 12, s, 64, s, False, True),
        ops.Linear(dtype, True, 1, n * s, 768, 768, False, False),
        ops.Linear(dtype, True, 1, n * s, 3072, 768, False, False),
        ops.Linear(dtype, True, 1, n * s, 768, 3072, False, False),
    ] * 12)

def bertbase384(dtype, n=1, s=384): return bertbase(dtype, n, s)

def resnet50(dtype, n=1):
    return Trace([
        ops.Conv2D(dtype, True, n, 224, 224, 3, 112, 112, 64, 7, 7, 2, 3, False),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 64, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 256, 56, 56, 64, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 256, 56, 56, 64, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 256, 56, 56, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 128, 28, 28, 128, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 512, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 56, 56, 256, 28, 28, 512, 1, 1, 2, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 28, 28, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 512, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 28, 28, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 512, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 28, 28, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 512, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 28, 28, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 256, 14, 14, 256, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 14, 14, 1024, 1, 1, 2, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 512, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 512, 7, 7, 512, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 2048, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 7, 7, 2048, 1, 1, 2, 0, False),
        ops.Conv2D(dtype, True, n, 7, 7, 2048, 7, 7, 512, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 512, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 2048, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 7, 7, 2048, 7, 7, 512, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 512, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 2048, 1, 1, 1, 0, False),
    ])

def ssdrn34_1200(dtype, n=1):
    return Trace([
        ops.Conv2D(dtype, True, n, 1200, 1200, 3, 600, 600, 64, 7, 7, 2, 3, False),
        ops.Conv2D(dtype, True, n, 300, 300, 64, 300, 300, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 300, 300, 64, 300, 300, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 300, 300, 64, 300, 300, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 300, 300, 64, 300, 300, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 300, 300, 64, 300, 300, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 300, 300, 64, 300, 300, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 300, 300, 64, 150, 150, 128, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 300, 300, 64, 150, 150, 128, 1, 1, 2, 0, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 128, 150, 150, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 150, 150, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 75, 75, 512, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 512, 75, 75, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 75, 75, 256, 38, 38, 512, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 512, 38, 38, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 19, 19, 256, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 19, 19, 256, 19, 19, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 19, 19, 128, 9, 9, 256, 3, 3, 2, 0, False),
        ops.Conv2D(dtype, True, n, 9, 9, 256, 9, 9, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 9, 9, 128, 7, 7, 256, 3, 3, 1, 0, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 50, 50, 16, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 150, 150, 256, 50, 50, 324, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 512, 25, 25, 24, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 512, 25, 25, 486, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 512, 13, 13, 24, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 512, 13, 13, 486, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 19, 19, 256, 7, 7, 24, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 19, 19, 256, 7, 7, 486, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 9, 9, 256, 3, 3, 16, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 9, 9, 256, 3, 3, 324, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 7, 7, 256, 3, 3, 16, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 7, 7, 256, 3, 3, 324, 3, 3, 3, 1, False),
    ])

def ssdrn34_300(dtype, n=1):
    return Trace([
        ops.Conv2D(dtype, True, n, 300, 300, 3, 150, 150, 64, 7, 7, 2, 3, False),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 38, 38, 128, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 38, 38, 128, 1, 1, 2, 0, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 19, 19, 512, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 19, 19, 512, 19, 19, 256, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 19, 19, 256, 10, 10, 512, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 10, 10, 512, 10, 10, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 10, 10, 128, 5, 5, 256, 3, 3, 2, 1, False),
        ops.Conv2D(dtype, True, n, 5, 5, 256, 5, 5, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 5, 5, 128, 3, 3, 256, 3, 3, 1, 0, False),
        ops.Conv2D(dtype, True, n, 3, 3, 256, 3, 3, 128, 1, 1, 1, 0, False),
        ops.Conv2D(dtype, True, n, 3, 3, 128, 1, 1, 256, 3, 3, 1, 0, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 13, 13, 16, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 13, 13, 324, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 19, 19, 512, 7, 7, 24, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 19, 19, 512, 7, 7, 486, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 10, 10, 512, 4, 4, 24, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 10, 10, 512, 4, 4, 486, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 5, 5, 256, 2, 2, 24, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 5, 5, 256, 2, 2, 486, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 3, 3, 256, 1, 1, 16, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 3, 3, 256, 1, 1, 324, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 1, 1, 256, 1, 1, 16, 3, 3, 3, 1, False),
        ops.Conv2D(dtype, True, n, 1, 1, 256, 1, 1, 324, 3, 3, 3, 1, False),
    ])

def unet3d(dtype, n=1):
    return Trace([
        ops.Conv3D(dtype, True, n, 128, 128, 128, 1, 128, 128, 128, 32, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 128, 128, 128, 32, 128, 128, 128, 32, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 128, 128, 128, 32, 64, 64, 64, 64, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 64, 64, 64, 64, 64, 64, 64, 64, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 64, 64, 64, 64, 32, 32, 32, 128, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 32, 32, 32, 128, 32, 32, 32, 128, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 32, 32, 32, 128, 16, 16, 16, 256, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 16, 16, 16, 256, 16, 16, 16, 256, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 16, 16, 16, 256, 8, 8, 8, 320, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 8, 8, 8, 320, 8, 8, 8, 320, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 8, 8, 8, 320, 4, 4, 4, 320, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 4, 4, 4, 320, 4, 4, 4, 320, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 8, 8, 8, 640, 8, 8, 8, 320, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 8, 8, 8, 320, 8, 8, 8, 320, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 16, 16, 16, 512, 16, 16, 16, 256, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 16, 16, 16, 256, 16, 16, 16, 256, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 32, 32, 32, 256, 32, 32, 32, 128, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 32, 32, 32, 128, 32, 32, 32, 128, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 64, 64, 64, 128, 64, 64, 64, 64, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 64, 64, 64, 64, 64, 64, 64, 64, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 128, 128, 128, 64, 128, 128, 128, 32, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 128, 128, 128, 32, 128, 128, 128, 32, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 128, 128, 128, 32, 128, 128, 128, 3, 1, 1, 1, 1, 0, False),
    ])

def rnnt_infer(dtype, n, il=200, ol=200):
    return Trace(([
        # Encoder
        ops.Lstm(dtype, False, n, 1, 240, 1024, False, False),
        ops.Lstm(dtype, False, n, 1, 1024, 1024, False, False),
    ] * il) + ([
        ops.Lstm(dtype, False, n, 1, 2048, 1024, False, False),
        ops.Lstm(dtype, False, n, 1, 1024, 1024, False, False),
        ops.Lstm(dtype, False, n, 1, 1024, 1024, False, False)
    ] * (il // 2)) + ([
        # Decoder
        ops.Lstm(dtype, False, n, 1, 320, 320, False, False),
        ops.Lstm(dtype, False, n, 1, 320, 320, False, False),
        ops.Linear(dtype, False, 1, n, 1344, 512, False, False),
        ops.Linear(dtype, False, 1, n, 512, 28, False, False),
    ] * ol))

def rnnt_train(dtype, n, il=200, ol=200):
    return Trace(([
        # Encoder
        ops.Lstm(dtype, True, n, 1, 240, 1024, False, False),
        ops.Lstm(dtype, True, n, 1, 1024, 1024, False, False),
    ] * il) + ([
        ops.Lstm(dtype, True, n, 1, 2048, 1024, False, False),
        ops.Lstm(dtype, True, n, 1, 1024, 1024, False, False),
        ops.Lstm(dtype, True, n, 1, 1024, 1024, False, False)
    ] * (il // 2)) + ([
        # Decoder
        ops.Lstm(dtype, True, n, 1, 320, 320, False, False),
        ops.Lstm(dtype, True, n, 1, 320, 320, False, False),
        ops.Linear(dtype, True, 1, n, 1344, 512, False, False),
        ops.Linear(dtype, True, 1, n, 512, 28, False, False),
    ] * ol))

def dlrm(dtype, n, tok_per_samp_frac=0.2):
    return Trace([
        # Dense Feature Layers
        ops.Linear(dtype, True, 1, n, 512, 13, False, False),
        ops.Linear(dtype, True, 1, n, 256, 512, False, False),
        ops.Linear(dtype, True, 1, n, 128, 256, False, False),

        # Sparse Feature Layers
        ops.Embedding(dtype, True, n, 39884406, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 39043, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 17289, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 7420, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 20263, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 3, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 7120, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 1543, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 63, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 38532951, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 2953546, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 403346, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 10, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 2208, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 11938, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 155, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 4, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 976, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 14, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 39979771, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 25641295, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 39664984, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 585935, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 12972, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 108, 128, tok_per_samp_frac),
        ops.Embedding(dtype, True, n, 36, 128, tok_per_samp_frac),

        # Interact Features is just a batch matmul
        ops.Matmul(dtype, True, n, 27, 27, 128, False, False),

        # Top MLP
        ops.Linear(dtype, True, 1, n, 479, 479, False, False),
        ops.Linear(dtype, True, 1, n, 1024, 479, False, False),
        ops.Linear(dtype, True, 1, n, 1024, 1024, False, False),
        ops.Linear(dtype, True, 1, n, 512, 1024, False, False),
        ops.Linear(dtype, True, 1, n, 256, 512, False, False),
        ops.Linear(dtype, True, 1, n, 1, 256, False, False),
    ])

app_infer_flops = {
    'bert-large-squad-avg': bertlarge(Dtype.I8, 1, 178).flops,
    'resnet50': resnet50(Dtype.I8, 1).flops,
    'unet': unet3d(Dtype.I8, 1).flops,
    'ssdrn34-1200': ssdrn34_1200(Dtype.I8, 1).flops,
    'rnnt': rnnt_infer(Dtype.I8, 1).flops,
    'dlrm': dlrm(Dtype.I8, 1).flops,
}

infer_apps_by_name = {
    'testconv': testconv_fwd,
    'unet': unet3d,
    'bert-large-squad-384': bertlarge384,
    'bert-large-squad-avg': bertlarge_squadavg,
    'bert-base-squad-384': bertbase384,
    'resnet50': resnet50,
    'ssdrn34-1200': ssdrn34_1200,
    'rnnt': rnnt_infer,
    'dlrm': dlrm
}

infer_batch_sizes = {
    'online': {
        'testconv': 1,
        'unet': 1,
        'bert-large-squad-avg': 1,
        'resnet50': 1,
        'ssdrn34-1200': 1,
        'rnnt': 1,
        'dlrm': 1,
    },
    'offline': {
        'testconv': 8,
        'unet': 8,
        'bert-large-squad-avg': 8,
        'resnet50': 8,
        'ssdrn34-1200': 4,
        'rnnt': 512,
        'dlrm': 1,
    },
}

infer_dtype = {
    'testconv': Dtype.I8,
    'unet': Dtype.I8,
    'bert-large-squad-avg': Dtype.I8,
    'resnet50': Dtype.I8,
    'ssdrn34-1200': Dtype.I8,
    'rnnt': Dtype.FP16,
    'dlrm': Dtype.I8,
}

app_train_flops = {
    'bert-large-512': bertlarge(Dtype.FP16, 1, 512).train().flops,
    'resnet50': resnet50(Dtype.FP16, 1).train().flops,
    'ssdrn34-300': ssdrn34_300(Dtype.FP16, 1).train().flops,
    'rnnt': rnnt_train(Dtype.FP16, 1).train().flops,
    'unet': unet3d(Dtype.FP16, 1).train().flops,
    'dlrm': dlrm(Dtype.FP16, 1).train().flops,
}

train_apps_by_name = {
    'testconv': testconv_bwd,
    'unet': unet3d,
    'bert-large-512': bertlarge,
    'resnet50': resnet50,
    'ssdrn34-300': ssdrn34_300,
    'rnnt': rnnt_train,
    'dlrm': dlrm
}

train_batch_sizes = {
    'small': {
        'testconv': 1,
        'unet': 1,
        'bert-large-512': 1,
        'resnet50': 1,
        'ssdrn34-1200': 1,
        'rnnt': 1,
        'dlrm': 1,
    },
    'large': {
        'testconv': 8,
        'unet': 8,
        'bert-large-512': 8,
        'resnet50': 4,
        'ssdrn34-1200': 4,
        'rnnt': 512,
        'dlrm': 1,
    },
}

train_dtype = {
    'testconv': Dtype.FP16,
    'bert-large-512': Dtype.FP16,
    'resnet50': Dtype.FP16,
    'ssdrn34-1200': Dtype.FP16,
    'rnnt': Dtype.FP16,
    'dlrm': Dtype.FP16,
}
