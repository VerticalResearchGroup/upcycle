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

    def train(self):
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

        new_list = self.oplist.copy()
        for op in reversed(self.oplist):
            if op.train:
                if type(op) in ops.backward_map:
                    backward_ops = ops.backward_map[type(op)]
                    for bop in backward_ops:
                        new_list.append(bop.from_forward(op))

                else:
                    logging.warn(f'No backward operator for {type(op)}. Ignoring.')

        self.oplist = new_list

    @property
    def unique_ops(self):
        return set(self.oplist)

def testmatmul(dtype, n=1):
    return Trace([ ops.Linear(dtype, True, 1, n, 1024, 1024, False, False) ])

def testconv_fwd(dtype, n=1):
    return Trace([ ops.Conv2D(dtype, True, n, 224, 224, 3, 112, 112, 64, 7, 7, 2, 3, False) ])

def testconv_bwd(dtype, n=1):
    return Trace([ ops.Conv2DBwd(dtype, True, n, 224, 224, 3, 112, 112, 64, 7, 7, 2, 3, False) ])

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
        ops.Conv3D(dtype, True, n, 224, 224, 224, 1, 224, 224, 224, 32, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 224, 224, 224, 32, 224, 224, 224, 32, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 224, 224, 224, 32, 112, 112, 112, 64, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 112, 112, 112, 64, 112, 112, 112, 64, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 112, 112, 112, 64, 56, 56, 56, 128, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 56, 56, 56, 128, 56, 56, 56, 128, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 56, 56, 56, 128, 28, 28, 28, 256, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 28, 28, 28, 256, 28, 28, 28, 256, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 28, 28, 28, 256, 14, 14, 14, 320, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 14, 14, 14, 320, 14, 14, 14, 320, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 14, 14, 14, 320, 7, 7, 7, 320, 3, 3, 3, 2, 1, False),
        ops.Conv3D(dtype, True, n, 7, 7, 7, 320, 7, 7, 7, 320, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 14, 14, 14, 640, 14, 14, 14, 320, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 14, 14, 14, 320, 14, 14, 14, 320, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 28, 28, 28, 512, 28, 28, 28, 256, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 28, 28, 28, 256, 28, 28, 28, 256, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 56, 56, 56, 256, 56, 56, 56, 128, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 56, 56, 56, 128, 56, 56, 56, 128, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 112, 112, 112, 128, 112, 112, 112, 64, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 112, 112, 112, 64, 112, 112, 112, 64, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 224, 224, 224, 64, 224, 224, 224, 32, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 224, 224, 224, 32, 224, 224, 224, 32, 3, 3, 3, 1, 1, False),
        ops.Conv3D(dtype, True, n, 224, 224, 224, 32, 224, 224, 224, 3, 1, 1, 1, 1, 0, False),
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
        ops.Lstm(dtype, False, n, 1, 240, 1024, False, False),
        ops.Lstm(dtype, False, n, 1, 1024, 1024, False, False),
    ] * il) + ([
        ops.Lstm(dtype, False, n, 1, 2048, 1024, False, False),
        ops.Lstm(dtype, False, n, 1, 1024, 1024, False, False),
        ops.Lstm(dtype, False, n, 1, 1024, 1024, False, False)
    ] * il // 2) + ([
        # Decoder
        ops.Lstm(dtype, False, n, 1, 320, 320, False, False),
        ops.Lstm(dtype, False, n, 1, 320, 320, False, False),
        ops.Linear(dtype, False, 1, n, 1344, 512, False, False),
        ops.Linear(dtype, False, 1, n, 512, 28, False, False),
    ] * ol))

app_infer_flops = {
    'bert-large-squad-avg': bertlarge(Dtype.I8, 1, 178).flops,
    'resnet50': resnet50(Dtype.I8, 1).flops,
    'ssdrn34-1200': ssdrn34_1200(Dtype.I8, 1).flops,
    'rnnt': rnnt_infer(Dtype.I8, 1).flops,
}

infer_apps_by_name = {
    'testconv': testconv_fwd,
    'bert-large-squad-384': bertlarge384,
    'bert-large-squad-avg': bertlarge_squadavg,
    'bert-base-squad-384': bertbase384,
    'resnet50': resnet50,
    'ssdrn34-1200': ssdrn34_1200,
    'rnnt': rnnt_infer,
}

infer_batch_sizes = {
    'online': {
        'testconv': 1,
        'bert-large-squad-avg': 1,
        'resnet50': 1,
        'ssdrn34-1200': 1,
        'rnnt': 1,
    },
    'offline': {
        'testconv': 8,
        'bert-large-squad-avg': 8,
        'resnet50': 4,
        'ssdrn34-1200': 4,
        'rnnt': 512,
    },
}

infer_dtype = {
    'testconv': Dtype.I8,
    'bert-large-squad-avg': Dtype.I8,
    'resnet50': Dtype.I8,
    'ssdrn34-1200': Dtype.I8,
    'rnnt': Dtype.FP16,
}


train_apps_by_name = {
    'testconv': testconv_bwd,
    # 'bert-large-squad-384': bertlarge384,
    # 'bert-base-squad-384': bertbase384,
    'resnet50': resnet50,
    'ssdrn34-300': ssdrn34_300,
    'rnnt': rnnt_train,
}

train_batch_sizes = {
    'small': {
        'testconv': 1,
        'bert-large-squad-avg': 1,
        'resnet50': 1,
        'ssdrn34-1200': 1,
        'rnnt': 1,
    },
    'large': {
        'testconv': 8,
        'bert-large-squad-avg': 8,
        'resnet50': 4,
        'ssdrn34-1200': 4,
        'rnnt': 512,
    },
}

train_dtype = {
    'testconv': Dtype.FP16,
    'bert-large-squad-avg': Dtype.FP16,
    'resnet50': Dtype.FP16,
    'ssdrn34-1200': Dtype.FP16,
    'rnnt': Dtype.FP16,
}
