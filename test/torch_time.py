import upcycle as U
import sys
import functools
import numpy as np
import logging
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an application')
    parser.add_argument('-d', '--dtype', type=str, default='')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-T', '--bwd-only', action='store_true')
    parser.add_argument('-i', '--infer', action='store_true')
    parser.add_argument('-a', '--app', type=str, default='resnet50')
    parser.add_argument('-b', '--batch', type=str, default='1')

    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('-l', '--layer', type=int, default=None)


    args = parser.parse_args()
    assert not (args.train and args.infer)
    assert args.train or args.infer

    if args.debug:
        assert args.parallel == 1, f'Cannot debug in parallel'
        logger.setLevel(logging.DEBUG)

    if args.infer:
        if args.batch in {'offline', 'online'}:
            batch = U.apps.infer_batch_sizes[args.batch][args.app]
        else: batch = int(args.batch)

        if args.dtype == '':
            dtype = U.apps.infer_dtype[args.app]
        else: dtype = U.Dtype.from_str(args.dtype)

        app = U.apps.infer_apps_by_name[args.app](dtype, n=batch)
        if args.layer is not None: app = U.apps.Trace([app.oplist[args.layer]])
        app.infer()

    else:
        if args.batch in {'large', 'small'}:
            batch = U.apps.train_batch_sizes[args.batch][args.app]
        else: batch = int(args.batch)

        if args.dtype == '':
            dtype = U.apps.train_dtype[args.app]
        else: dtype = U.Dtype.from_str(args.dtype)

        app = U.apps.train_apps_by_name[args.app](dtype, n=batch)
        if args.layer is not None: app = U.apps.Trace([app.oplist[args.layer]])
        app.train(args.bwd_only)

    logging.info(f'App: {args.app} ({"train" if args.train else "infer"})')
    logging.info(f'Dtype: {dtype}, Batch Size: {batch}')
    logging.info(f'App Ops: {app.flops / 1e9} G')

    a100_peak = U.nvdb.a100_peak[dtype]
    times = U.torchrun.run_with_torch(app, device_type='cuda', niters=100)

    logger.info(f'{blue}Times: {times} {reset}')
    total_time = np.sum(times)
    logger.info(f'{green}Total Time: {total_time} {reset}')
    logger.info(f'GOPS: {app.flops / total_time / 1e9}')
    logger.info(f'A100 Utilization: {app.flops / total_time / a100_peak * 100} %')

