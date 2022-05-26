from dataclasses import dataclass

from . import ops

@dataclass
class Trace:
    oplist : list[ops.Operator]

    @property
    def flops(self): return sum(op.flops for op in self.oplist)

    def train(self):
        new_list = self.oplist.copy()
        for op in reversed(self.oplist):
            if op.train:
                backward_ops = ops.backward_map[type(op)]
                for bop in backward_ops:
                    new_list.append(bop.from_forward(op))

        self.oplist = new_list

def bertlarge(dtype, n=1, s=512):
    return Trace([
        ops.Matmul(dtype, True, 1, n * s, 1024, 1024, False, False),
        ops.Matmul(dtype, True, 1, n * s, 1024, 1024, False, False),
        ops.Matmul(dtype, True, 1, n * s, 1024, 1024, False, False),
        ops.Matmul(dtype, True, n * 16, s, s, 64, False, False),
        # TODO: Softmax
        ops.Matmul(dtype, True, n * 16, s, 64, s, False, False),
        ops.Matmul(dtype, True, 1, n * s, 1024, 1024, False, False),
        ops.Matmul(dtype, True, 1, n * s, 4096, 1024, False, False),
        ops.Matmul(dtype, True, 1, n * s, 1024, 4096, False, False),
    ] * 24)

def bertbase(dtype, n=1, s=512):
    return Trace([
        ops.Matmul(dtype, True, 1, n * s, 768, 768, False, False),
        ops.Matmul(dtype, True, 1, n * s, 768, 768, False, False),
        ops.Matmul(dtype, True, 1, n * s, 768, 768, False, False),
        ops.Matmul(dtype, True, n * 12, s, s, 64, False, False),
        # TODO: Softmax
        ops.Matmul(dtype, True, n * 12, s, 64, s, False, False),
        ops.Matmul(dtype, True, 1, n * s, 768, 768, False, False),
        ops.Matmul(dtype, True, 1, n * s, 3072, 768, False, False),
        ops.Matmul(dtype, True, 1, n * s, 768, 3072, False, False),
    ] * 12)

def resnet50(dtype, n=1):
    return Trace([
        ops.Conv2D(dtype, True, n, 224, 224, 3, 112, 112, 64, 7, 7, 2),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 64, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 256, 56, 56, 64, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 256, 56, 56, 64, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 64, 56, 56, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 256, 56, 56, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 128, 28, 28, 128, 3, 3, 2),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 512, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 56, 56, 256, 28, 28, 512, 1, 1, 2),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 28, 28, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 512, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 28, 28, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 512, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 28, 28, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 128, 28, 28, 512, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 28, 28, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 256, 14, 14, 256, 3, 3, 2),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 28, 28, 512, 14, 14, 1024, 1, 1, 2),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 256, 14, 14, 1024, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 14, 14, 512, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 512, 7, 7, 512, 3, 3, 2),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 2048, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 14, 14, 1024, 7, 7, 2048, 1, 1, 2),
        ops.Conv2D(dtype, True, n, 7, 7, 2048, 7, 7, 512, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 512, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 2048, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 7, 7, 2048, 7, 7, 512, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 512, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 7, 7, 512, 7, 7, 2048, 1, 1, 1),
    ])

def ssdrn34_1200(dtype, n=1):
    return Trace([
        ops.Conv2D(dtype, True, 1, 1200, 1200, 3, 600, 600, 64, 7, 7, 2),
        ops.Conv2D(dtype, True, 1, 300, 300, 64, 300, 300, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 300, 300, 64, 300, 300, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 300, 300, 64, 300, 300, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 300, 300, 64, 300, 300, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 300, 300, 64, 300, 300, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 300, 300, 64, 300, 300, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 300, 300, 64, 150, 150, 128, 3, 3, 2),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 300, 300, 64, 150, 150, 128, 1, 1, 2),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 128, 150, 150, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 150, 150, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 75, 75, 512, 3, 3, 2),
        ops.Conv2D(dtype, True, 1, 75, 75, 512, 75, 75, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, 1, 75, 75, 256, 38, 38, 512, 3, 3, 2),
        ops.Conv2D(dtype, True, 1, 38, 38, 512, 38, 38, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, 1, 38, 38, 128, 19, 19, 256, 3, 3, 2),
        ops.Conv2D(dtype, True, 1, 19, 19, 256, 19, 19, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, 1, 19, 19, 128, 9, 9, 256, 3, 3, 2),
        ops.Conv2D(dtype, True, 1, 9, 9, 256, 9, 9, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, 1, 9, 9, 128, 7, 7, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 50, 50, 16, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 150, 150, 256, 50, 50, 324, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 75, 75, 512, 25, 25, 24, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 75, 75, 512, 25, 25, 486, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 38, 38, 512, 13, 13, 24, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 38, 38, 512, 13, 13, 486, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 19, 19, 256, 7, 7, 24, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 19, 19, 256, 7, 7, 486, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 9, 9, 256, 3, 3, 16, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 9, 9, 256, 3, 3, 324, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 7, 7, 256, 3, 3, 16, 3, 3, 3),
        ops.Conv2D(dtype, True, 1, 7, 7, 256, 3, 3, 324, 3, 3, 3),
    ])

def ssdrn34_300(dtype, n=1):
    return Trace([
        ops.Conv2D(dtype, True, n, 300, 300, 3, 150, 150, 64, 7, 7, 2),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 75, 75, 64, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 38, 38, 128, 3, 3, 2),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 75, 75, 64, 38, 38, 128, 1, 1, 2),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 128, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 128, 38, 38, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 19, 19, 512, 3, 3, 2),
        ops.Conv2D(dtype, True, n, 19, 19, 512, 19, 19, 256, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 19, 19, 256, 10, 10, 512, 3, 3, 2),
        ops.Conv2D(dtype, True, n, 10, 10, 512, 10, 10, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 10, 10, 128, 5, 5, 256, 3, 3, 2),
        ops.Conv2D(dtype, True, n, 5, 5, 256, 5, 5, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 5, 5, 128, 3, 3, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 3, 3, 256, 3, 3, 128, 1, 1, 1),
        ops.Conv2D(dtype, True, n, 3, 3, 128, 1, 1, 256, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 16, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 38, 38, 256, 38, 38, 324, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 19, 19, 512, 19, 19, 24, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 19, 19, 512, 19, 19, 486, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 10, 10, 512, 10, 10, 24, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 10, 10, 512, 10, 10, 486, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 5, 5, 256, 5, 5, 24, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 5, 5, 256, 5, 5, 486, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 3, 3, 256, 3, 3, 16, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 3, 3, 256, 3, 3, 324, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 1, 1, 256, 1, 1, 16, 3, 3, 1),
        ops.Conv2D(dtype, True, n, 1, 1, 256, 1, 1, 324, 3, 3, 1),
    ])

