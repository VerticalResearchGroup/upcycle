import upcycle as U
import sys
import functools
import numpy as np
import logging
import multiprocessing
import time
import argparse
import tqdm

blue = '\x1b[38;5;39m'
green = '\033[92m'
reset = '\x1b[0m'

logging._srcfile = None
logger = logging.getLogger()
ch = logging.StreamHandler(sys.stdout)
fmt = U.logutils.CustomFormatter()
ch.setFormatter(fmt)
logging.basicConfig(level=logging.INFO, handlers=[ch])
logger.setLevel(logging.DEBUG)


parser = argparse.ArgumentParser(description='Simulate an application')
U.arch.arch_cli_params(parser)
args = parser.parse_args()

arch = U.arch.arch_from_cli(args)
conv = U.ops.Conv2D(U.Dtype.FP16, False, 8, 224, 224, 3, 112, 112, 64, 7, 7, 2, 3, False, False)

tdi, tw, tdo = U.ops.conv2d.make_conv2d_tensors(arch, conv)
print(tdi.shape, tw.shape, tdo.shape)

tile = U.ops.conv2d_di.Conv2DDiTile(
    arch, conv, [tdo, tw], [tdi], False,
    U.Slice(0, 1),
    U.Slice(conv.pad, conv.pad + conv.stride),
    U.Slice(conv.pad, conv.pad + conv.stride),
    U.Slice(0, 2),
    U.Slice(0, 3))

print(tile)

print('Latency: ', tile.exec_lat)
print('Perfect Latency: ', tile.perfect_exec_lat)
print('Flops: ', tile.flops)
print('% of Perfect: ', tile.exec_lat / tile.perfect_exec_lat)
