from dataclasses import dataclass
from typing import Callable

from .common import *
from . import ops

@dataclass
class Trace:
    """A trace of DL Operators."""
    bs : int
    oplist : list[ops.Operator]

    @property
    def flops(self): return sum(op.flops for op in self.oplist)

    def infer(self):
        """Setup this trace for Inference.

        This applies transformations which optimize certain Operators for
        inference (E.g. transposing weights for Linear layers).
        """
        for i, op in enumerate(self.oplist):
            if type(op) is ops.Linear:
                self.oplist[i] = ops.Linear(
                    op.dtype, op.train, op.l, op.m, op.n, op.k, False, True)

            elif type(op) is ops.LstmCell:
                self.oplist[i] = ops.LstmCell(
                    op.dtype, op.train, op.n, op.d, op.h, False, True)

        return self

    def train(self, bwd_only=False):
        """Setup this trace for Training.

        This function will find and append backward operations to the trace for
        every trainable forward operator. It will also apply transformations to
        optimize operators for training performance.
        """
        for i, op in enumerate(self.oplist):
            if type(op) is ops.Linear:
                self.oplist[i] = ops.Linear(
                    op.dtype, op.train, op.l, op.m, op.n, op.k, False, False)

            elif type(op) is ops.LstmCell:
                self.oplist[i] = ops.LstmCell(
                    op.dtype, op.train, op.n, op.d, op.h, False, False)

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

    def count(self, op):
        return self.oplist.count(op)

def testmatmul(dtype, n=1):
    return Trace(n, [ ops.Linear(dtype, True, 1, n, 1024, 1024, False, False) ])

def testconv(dtype, n=1):
    return Trace(n, [ ops.Conv2D(dtype, True, n, 224, 224, 3, 112, 112, 64, 7, 7, 2, 3, False) ])

# MLPerf Apps based on v2.0 Inference and v1.1 Training datacenter benchmarks:
# https://mlcommons.org/en/inference-datacenter-20/


def bertlarge(dtype, n=1, s=512):
    return Trace(n, [
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
    return Trace(n, [
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
    return Trace(n, [
        ops.Conv(dtype, True, n, (224, 224), 3, (112, 112), 64, (7, 7), 2, 3, False),
        ops.Conv(dtype, True, n, (56, 56), 64, (56, 56), 64, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 64, (56, 56), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (56, 56), 64, (56, 56), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 64, (56, 56), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 256, (56, 56), 64, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 64, (56, 56), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (56, 56), 64, (56, 56), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 256, (56, 56), 64, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 64, (56, 56), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (56, 56), 64, (56, 56), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 256, (56, 56), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 128, (28, 28), 128, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (28, 28), 128, (28, 28), 512, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (56, 56), 256, (28, 28), 512, (1, 1), 2, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 512, (28, 28), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 128, (28, 28), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (28, 28), 128, (28, 28), 512, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 512, (28, 28), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 128, (28, 28), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (28, 28), 128, (28, 28), 512, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 512, (28, 28), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 128, (28, 28), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (28, 28), 128, (28, 28), 512, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 512, (28, 28), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 256, (14, 14), 256, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 1024, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (28, 28), 512, (14, 14), 1024, (1, 1), 2, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 1024, (14, 14), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 1024, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 1024, (14, 14), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 1024, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 1024, (14, 14), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 1024, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 1024, (14, 14), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 1024, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 1024, (14, 14), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (14, 14), 256, (14, 14), 1024, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 1024, (14, 14), 512, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 512, (7, 7), 512, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (7, 7), 512, (7, 7), 2048, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (14, 14), 1024, (7, 7), 2048, (1, 1), 2, 0, False),
        ops.Conv(dtype, True, n, (7, 7), 2048, (7, 7), 512, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (7, 7), 512, (7, 7), 512, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (7, 7), 512, (7, 7), 2048, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (7, 7), 2048, (7, 7), 512, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (7, 7), 512, (7, 7), 512, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (7, 7), 512, (7, 7), 2048, (1, 1), 1, 0, False),
    ])

def ssdrn34_1200(dtype, n=1):
    return Trace(n, [
        ops.Conv(dtype, True, n, (1200, 1200), 3, (600, 600), 64, (7, 7), 2, 3, False),
        ops.Conv(dtype, True, n, (300, 300), 64, (300, 300), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (300, 300), 64, (300, 300), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (300, 300), 64, (300, 300), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (300, 300), 64, (300, 300), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (300, 300), 64, (300, 300), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (300, 300), 64, (300, 300), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (300, 300), 64, (150, 150), 128, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (300, 300), 64, (150, 150), 128, (1, 1), 2, 0, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 128, (150, 150), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (150, 150), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (75, 75), 512, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 512, (75, 75), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (75, 75), 256, (38, 38), 512, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 512, (38, 38), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (19, 19), 256, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (19, 19), 256, (19, 19), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (19, 19), 128, (9, 9), 256, (3, 3), 2, 0, False),
        ops.Conv(dtype, True, n, (9, 9), 256, (9, 9), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (9, 9), 128, (7, 7), 256, (3, 3), 1, 0, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (50, 50), 16, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (150, 150), 256, (50, 50), 324, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 512, (25, 25), 24, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 512, (25, 25), 486, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 512, (13, 13), 24, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 512, (13, 13), 486, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (19, 19), 256, (7, 7), 24, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (19, 19), 256, (7, 7), 486, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (9, 9), 256, (3, 3), 16, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (9, 9), 256, (3, 3), 324, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (7, 7), 256, (3, 3), 16, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (7, 7), 256, (3, 3), 324, (3, 3), 3, 1, False),
    ])

def ssdrn34_300(dtype, n=1):
    return Trace(n, [
        ops.Conv(dtype, True, n, (300, 300), 3, (150, 150), 64, (7, 7), 2, 3, False),
        ops.Conv(dtype, True, n, (75, 75), 64, (75, 75), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 64, (75, 75), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 64, (75, 75), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 64, (75, 75), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 64, (75, 75), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 64, (75, 75), 64, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 64, (38, 38), 128, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (75, 75), 64, (38, 38), 128, (1, 1), 2, 0, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 128, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 128, (38, 38), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (38, 38), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (19, 19), 512, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (19, 19), 512, (19, 19), 256, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (19, 19), 256, (10, 10), 512, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (10, 10), 512, (10, 10), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (10, 10), 128, (5, 5), 256, (3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (5, 5), 256, (5, 5), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (5, 5), 128, (3, 3), 256, (3, 3), 1, 0, False),
        ops.Conv(dtype, True, n, (3, 3), 256, (3, 3), 128, (1, 1), 1, 0, False),
        ops.Conv(dtype, True, n, (3, 3), 128, (1, 1), 256, (3, 3), 1, 0, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (13, 13), 16, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (38, 38), 256, (13, 13), 324, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (19, 19), 512, (7, 7), 24, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (19, 19), 512, (7, 7), 486, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (10, 10), 512, (4, 4), 24, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (10, 10), 512, (4, 4), 486, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (5, 5), 256, (2, 2), 24, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (5, 5), 256, (2, 2), 486, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (3, 3), 256, (1, 1), 16, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (3, 3), 256, (1, 1), 324, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (1, 1), 256, (1, 1), 16, (3, 3), 3, 1, False),
        ops.Conv(dtype, True, n, (1, 1), 256, (1, 1), 324, (3, 3), 3, 1, False),
    ])

def ssd_weirdconv(dtype, n=1):
    return Trace(n, [
        ops.Conv(dtype, True, n, (5, 5), 128, (3, 3), 256, (3, 3), 1, 0, False)
    ])

def unet3d(dtype, n=1):
    #
    # N.B. The first and last few layers of unet are 128x128x128 sized inputs /
    # outpus which make the layers quite large in terms of op-count footprint.
    # This takes a very long time to run simulations for. To help the model, I
    # use smaller spatial dimensions for these layers, and then multiply their
    # instance count by the factor reduction. This is semantically equivalent to
    # computing the convolution in stages over the spatial dimensions of the
    # input. It shouldn't affect the core utilization at all, and also shouldn't
    # meaninfgully impact the generated traffic, either.
    #
    # This makes simulation faster because the model will only simulate each
    # unique op once, reducing overall sim time.
    #

    return Trace(n, [
        # N.B. Original spatial dimensions are 128x128x128
        ops.Conv(dtype, True, n, (32, 32, 32), 1, (32, 32, 32), 32, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 32, (32, 32, 32), 32, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 32, (16, 16, 16), 64, (3, 3, 3), 2, 1, False),
    ] * (4*4*4) + [
        ops.Conv(dtype, True, n, (32, 32, 32), 64, (32, 32, 32), 64, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 64, (16, 16, 16), 128, (3, 3, 3), 2, 1, False),
    ] * (2*2*2) + [
        ops.Conv(dtype, True, n, (32, 32, 32), 128, (32, 32, 32), 128, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 128, (16, 16, 16), 256, (3, 3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (16, 16, 16), 256, (16, 16, 16), 256, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (16, 16, 16), 256, (8, 8, 8), 320, (3, 3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (8, 8, 8), 320, (8, 8, 8), 320, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (8, 8, 8), 320, (4, 4, 4), 320, (3, 3, 3), 2, 1, False),
        ops.Conv(dtype, True, n, (4, 4, 4), 320, (4, 4, 4), 320, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (8, 8, 8), 640, (8, 8, 8), 320, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (8, 8, 8), 320, (8, 8, 8), 320, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (16, 16, 16), 512, (16, 16, 16), 256, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (16, 16, 16), 256, (16, 16, 16), 256, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 256, (32, 32, 32), 128, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 128, (32, 32, 32), 128, (3, 3, 3), 1, 1, False),
    ] + [
        ops.Conv(dtype, True, n, (32, 32, 32), 128, (32, 32, 32), 64, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 64, (32, 32, 32), 64, (3, 3, 3), 1, 1, False),
    ] * (2*2*2) + [
        # N.B. Original spatial dimensions are 128x128x128
        ops.Conv(dtype, True, n, (32, 32, 32), 64, (32, 32, 32), 32, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 32, (32, 32, 32), 32, (3, 3, 3), 1, 1, False),
        ops.Conv(dtype, True, n, (32, 32, 32), 32, (32, 32, 32), 3, (1, 1, 1), 1, 0, False),
    ] * (4*4*4))

# N.B. The official Kits19 dataset for unet runs many forward passes per sample,
# averaging 1544696 fwds / 24612 samples = ~63 fwd passes / sample. Here we
# still use one forward pass as the unit of measurement, and scale NVIDIA's
# reported throughput by this factor.
kits19_patches_per_sample = 1544696 / 24612

def rnnt_infer(dtype, n, il=239, ol=120):
    return Trace(n, ([
        # Encoder
        ops.LstmCell(dtype, False, n, 240, 1024, False, False),
        ops.LstmCell(dtype, False, n, 1024, 1024, False, False),
    ] * il) + ([
        ops.LstmCell(dtype, False, n, 2048, 1024, False, False),
        ops.LstmCell(dtype, False, n, 1024, 1024, False, False),
        ops.LstmCell(dtype, False, n, 1024, 1024, False, False)
    ] * (il // 2)) + ([
        # Decoder
        ops.LstmCell(dtype, False, n, 320, 320, False, False),
        ops.LstmCell(dtype, False, n, 320, 320, False, False),
        ops.Linear(dtype, False, 1, n, 1344, 512, False, False),
        ops.Linear(dtype, False, 1, n, 512, 28, False, False),
    ] * ol))

def rnnt_train(dtype, n, il=200, ol=200):
    return Trace(n, ([
        # Encoder
        ops.LstmCell(dtype, True, n, 240, 1024, False, False),
        ops.LstmCell(dtype, True, n, 1024, 1024, False, False),
    ] * il) + ([
        ops.LstmCell(dtype, True, n, 2048, 1024, False, False),
        ops.LstmCell(dtype, True, n, 1024, 1024, False, False),
        ops.LstmCell(dtype, True, n, 1024, 1024, False, False)
    ] * (il // 2)) + ([
        # Decoder
        ops.LstmCell(dtype, True, n, 320, 320, False, False),
        ops.LstmCell(dtype, True, n, 320, 320, False, False),
        ops.Linear(dtype, True, 1, n, 1344, 512, False, False),
        ops.Linear(dtype, True, 1, n, 512, 28, False, False),
    ] * ol))

def dlrm(dtype, n, tok_per_samp_frac=0.2):
    return Trace(n, [
        # Dense Feature Layers
        ops.Linear(dtype, True, 1, n, 512, 13, False, False),
        ops.Linear(dtype, True, 1, n, 256, 512, False, False),
        ops.Linear(dtype, True, 1, n, 128, 256, False, False),

        # Sparse Feature Layers
        # ops.Embedding(dtype, True, n, 39884406, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 39043, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 17289, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 7420, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 20263, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 3, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 7120, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 1543, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 63, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 38532951, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 2953546, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 403346, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 10, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 2208, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 11938, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 155, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 4, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 976, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 14, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 39979771, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 25641295, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 39664984, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 585935, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 12972, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 108, 128, tok_per_samp_frac),
        # ops.Embedding(dtype, True, n, 36, 128, tok_per_samp_frac),

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

@dataclass
class BatchSizes:
    infer_online : int
    infer_offline : int
    train_small : int
    train_large : int

@dataclass
class App:
    infer_factory : Callable[[], Trace]
    infer_dtype : Dtype
    train_factory : Callable[[], Trace]
    train_dtype : Dtype
    bs : BatchSizes

    @property
    def infer_flops(self):
        return self.infer_factory(self.infer_dtype, n=1).infer().flops

    @property
    def train_flops(self):
        return self.train_factory(self.train_dtype, n=1).train().flops

    def default_infer_online(self): return self.infer_factory(self.infer_dtype, n=self.bs.infer_online).infer()
    def default_infer_offline(self): return self.infer_factory(self.infer_dtype, n=self.bs.infer_offline).infer()
    def default_train_small(self): return self.train_factory(self.train_dtype, n=self.bs.train_small).train()
    def default_train_large(self): return self.train_factory(self.train_dtype, n=self.bs.train_large).train()

mlperf_v1_apps = {
    'testmm': App(
        testmatmul, Dtype.I8,
        testmatmul, Dtype.FP16,
        BatchSizes(1, 8, 1, 8)),
    'testconv': App(
        testconv, Dtype.I8,
        testconv, Dtype.FP16,
        BatchSizes(1, 8, 1, 8)),
    'ssd_weirdconv': App(
        ssd_weirdconv, Dtype.I8,
        ssd_weirdconv, Dtype.FP16,
        BatchSizes(1, 16, 1, 16)),
    'resnet50': App(
        resnet50, Dtype.I8,
        resnet50, Dtype.FP16,
        BatchSizes(1, 16, 1, 16)),
    'ssdrn34-300': App(
        None, None,
        ssdrn34_300, Dtype.FP16,
        BatchSizes(None, None, 1, 16)),
    'ssdrn34-1200': App(
        ssdrn34_1200, Dtype.I8,
        None, None,
        BatchSizes(1, 16, None, None)),
    'bert-large-squad': App(
        # N.B. 178 reflects the average tokens per query from the SQuAD dataset.
        lambda dtype, n: bertlarge(dtype, n, 178), Dtype.I8,
        None, None,
        BatchSizes(1, 16, None, None)),
    'bert-large-pretrain': App(
        None, None,
        # N.B. The official MLPerf code was run with bert pretraining for a full
        # epoch and we recorded among 156689001 total samples observed for
        # training, there was 39867330529 total tokens -- an average of 254
        # tokens per sample.
        lambda dtype, n: bertlarge(dtype, n, 254), Dtype.FP16,
        BatchSizes(None, None, 1, 4)),
    'unet': App(
        unet3d, Dtype.I8,
        unet3d, Dtype.FP16,
        BatchSizes(1, 2, 1, 16)),
    'rnnt': App(
        # N.B. The official MLPerf inference benchmark uses librespeech dataset
        # which has an average input length of 239, output length of 120.
        lambda dtype, n: rnnt_infer(dtype, n, il=239, ol=120), Dtype.FP16,
        rnnt_train, Dtype.FP16,
        BatchSizes(1, 512, 1, 512)),
    'rnnt-test': App(
        lambda dtype, n: rnnt_infer(dtype, n, il=389, ol=191), Dtype.FP16,
        rnnt_train, Dtype.FP16,
        BatchSizes(1, 512, 1, 512)),
    'dlrm': App(
        dlrm, Dtype.I8,
        dlrm, Dtype.FP16,
        BatchSizes(1, 2048, 1, 2048)),
}

short_appname_map = {
    ('resnet50', 'infer'): 'resnet50',
    ('resnet50', 'train'): 'resnet50',
    ('ssdrn34', 'infer'): 'ssdrn34-1200',
    ('ssdrn34', 'train'): 'ssdrn34-300',
    ('bert', 'infer'): 'bert-large-squad',
    ('bert', 'train'): 'bert-large-pretrain',
    ('unet', 'infer'): 'unet',
    ('unet', 'train'): 'unet',
    ('rnnt', 'infer'): 'rnnt',
    ('rnnt', 'train'): 'rnnt'
}

def workload_cli_params(parser):
    parser.add_argument('-d', '--dtype', type=str, default='')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-T', '--bwd-only', action='store_true')
    parser.add_argument('-i', '--infer', action='store_true')
    parser.add_argument('-a', '--app', type=str, default='resnet50')
    parser.add_argument('-b', '--batch', type=str, default='1')
    parser.add_argument('-m', '--placement-mode', type=str, default='pg')
    parser.add_argument('-l', '--layer', type=int, default=None)

def workload_factory(app, scenario_or_batch, infer=True, layer=None, bwd_only=False):
    app = mlperf_v1_apps[app]

    if infer:
        if scenario_or_batch in {'online', 'offline'}:
            batch = {
                'online': app.bs.infer_online,
                'offline': app.bs.infer_offline,
            }[scenario_or_batch]
        else: batch = int(scenario_or_batch)

        dtype = app.infer_dtype
        trace = app.infer_factory(dtype, n=batch)

        trace.infer()
        if layer is not None: trace = Trace(trace.bs, [trace.oplist[layer]])

    else:
        if scenario_or_batch in {'small', 'large'}:
            batch = {
                'small': app.bs.train_small,
                'large': app.bs.train_large,
            }[scenario_or_batch]
        else: batch = int(scenario_or_batch)

        dtype = app.train_dtype
        trace = app.train_factory(dtype, n=batch)

        trace.train(bwd_only)
        if layer is not None: trace = Trace(trace.bs, [trace.oplist[layer]])

    return app, trace, batch, dtype

def workload_from_cli(args):
    assert not (args.train and args.infer)
    assert args.train or args.infer
    return workload_factory(args.app, args.batch, args.infer, args.layer, args.bwd_only)
