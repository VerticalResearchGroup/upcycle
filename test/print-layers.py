import upcycle as U
import sys
import functools
import numpy as np
import logging
import multiprocessing
import time
import argparse
import tqdm
# import tracemalloc

blue = '\x1b[38;5;39m'
green = '\033[92m'
reset = '\x1b[0m'

logging._srcfile = None
logger = logging.getLogger()
ch = logging.StreamHandler(sys.stdout)
fmt = U.logutils.CustomFormatter()
ch.setFormatter(fmt)
fmt.keywords.append('RSS')
logging.basicConfig(level=logging.INFO, handlers=[ch])

# logging.getLogger('upcycle.ops.common').setLevel(logging.INFO)


def log_layer(arch : U.Arch, dtype : U.Dtype, app : U.apps.Trace, i, op : U.ops.Operator, steps : bool):
    if steps:
        nsteps = U.model.num_steps(arch, op)
        logger.info(f'{green}Layer {i}/{len(app.oplist)}: {op} {reset} ({nsteps} steps)')
    else:
        logger.info(f'{green}Layer {i}/{len(app.oplist)}: {op} {reset}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an application')

    U.arch.arch_cli_params(parser)
    U.apps.workload_cli_params(parser)

    parser.add_argument('-p', '--parallel', type=int, default=1)
    parser.add_argument('-s', '--steps', action='store_true')

    args = parser.parse_args()
    # logger.setLevel(logging.DEBUG)


    arch = U.arch.arch_from_cli(args)
    app, trace, batch, dtype = U.apps.workload_from_cli(args)

    logging.info(f'App: {args.app} ({"train" if args.train else "infer"})')
    logging.info(f'Dtype: {dtype}, Batch Size: {batch}')
    logging.info(f'App Ops: {trace.flops / 1e9} G')
    logging.info(f'Arch: {arch}')
    logging.info('=' * 40)

    for i, op in enumerate(trace.oplist):
        log_layer(arch, dtype, trace, i, op, args.steps)
