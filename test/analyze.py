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

@functools.singledispatch
def analyze(op):
    logger.warn('+ No information available')

@analyze.register(U.ops.Conv2D)
def _(op : U.ops.Conv2D):
    logger.info(f'+ # weight elements: {op.r * op.s}')
    logger.info(f'+ Channels: {op.c} -> {op.k}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze an application')
    U.apps.workload_cli_params(parser)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    app, trace, batch, dtype = U.apps.workload_from_cli(args)

    op : U.Operator
    for i, op in enumerate(trace.oplist):
        logger.info(f'{i}/{len(trace.oplist)}: {op}')
        analyze(op)
