import upcycle as U
import sys
import functools
import numpy as np
import logging
import multiprocessing
import time
import argparse

logging._srcfile = None
logger = logging.getLogger()
ch = logging.StreamHandler(sys.stdout)
fmt = U.logutils.CustomFormatter()
ch.setFormatter(fmt)
logging.basicConfig(level=logging.INFO, handlers=[ch])

def simulate_layer(arch : U.Arch, op : U.ops.Operator, sim_kwargs):
    logger.debug(f'Simulating {op}...')
    result = U.model.simulate(arch, op, **sim_kwargs)
    logger.debug(f'Finished {op}')
    return result

def log_layer(arch : U.Arch, app : U.apps.Trace, i, op : U.ops.Operator, result : U.model.SimResult, time_ns=None, details=True):
    blue = '\x1b[38;5;39m'
    green = '\033[92m'
    reset = '\x1b[0m'
    gops = int(op.flops / result.cycles * arch.freq / 1e9)
    eff = np.round(op.flops / result.cycles / arch.ntiles / arch.peak_opc(U.Dtype.I8) * 100, 2)
    logger.info(f'{green}Layer {i}/{len(app.oplist)}: {op} {reset}')
    if details:
        if time_ns is not None: logger.info(f'+ Simulation time: {time_ns / 1e9} s')
        logger.info(f'+ Latency: {int(result.cycles)} cyc, Hops: {np.sum(result.traffic)}')
        logger.info(f'+ Compute: {gops} Gops ({blue}Efficiency: {eff} %{reset})')
        if 'avg_dests' in result.kwstats:
            logger.info(f'+ (Max) Avg Dests Per Line: {np.max(result.kwstats["avg_dests"])}')
        if 'avg_groups' in result.kwstats:
            logger.info(f'+ (Max) Avg Groups Per Line: {np.max(result.kwstats["avg_groups"])}')


def simulate_app(arch : U.Arch, app : U.apps.Trace, sim_args):
    layers = []
    cache = {}
    tt0 = time.perf_counter_ns()
    for i, op in enumerate(app.oplist):
        t0 = time.perf_counter_ns()
        if op not in cache:
            result = simulate_layer(arch, op, sim_args)
            cache[op] = result
            hit = False
        else:
            result = cache[op]
            hit = True

        layers.append(result)

        t1 = time.perf_counter_ns()
        log_layer(arch, app, i, op, result, t1 - t0, details=not hit)

    tt1 = time.perf_counter_ns()
    return tt1 - tt0, layers

def simulate_app_par(arch : U.Arch, app : U.apps.Trace, sim_args):
    pool = multiprocessing.Pool(16)
    tt0 = time.perf_counter_ns()
    unique_ops = list(app.unique_ops)
    unique_results = pool.map(
        functools.partial(simulate_layer, arch, sim_kwargs=sim_args), unique_ops)

    cache = {op: result for op, result in zip(unique_ops, unique_results)}
    tt1 = time.perf_counter_ns()

    layers = []
    for i, op in enumerate(app.oplist):
        result = cache[op]
        layers.append(result)
        log_layer(arch, app, i, op, result)

    return tt1 - tt0, layers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an application')
    parser.add_argument('--arch', type=str, default='oracle')
    parser.add_argument('-d', '--dtype', type=str, default='I8')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-i', '--infer', action='store_true')
    parser.add_argument('-a', '--app', type=str, default='resnet50')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-m', '--placement-mode', type=str, default='pg')
    parser.add_argument('-p', '--parallel', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-l', '--layer', type=int, default=None)

    parser.add_argument('--noc-ports', type=int, default=1)
    parser.add_argument('--l1-capacity', type=int, default=64*1024)
    parser.add_argument('--l1-assoc', type=int, default=16)
    parser.add_argument('--bgsize', type=str, default='4,8')

    args = parser.parse_args()
    assert not (args.train and args.infer)
    assert args.train or args.infer

    if args.verbose:
        assert not args.parallel, f'Cannot debug in parallel'
        logger.setLevel(logging.DEBUG)

    if args.arch == 'oracle':
        arch = U.OracleArch(2.4e9, 512, 1, 32, 64, args.noc_ports)
    elif args.arch == 'bg':
        [grows, gcols] = list(map(int, args.bgsize.split(',')))
        arch = U.BgroupArch(2.4e9, 512, 1, 32, 64, args.noc_ports, grows, gcols)

    dtype = U.Dtype.from_str(args.dtype)

    if args.infer:
        app = U.apps.infer_apps_by_name[args.app](dtype, n=args.batch)
        if args.layer is not None: app = U.apps.Trace([app.oplist[args.layer]])
        app.infer()

    else:
        app = U.apps.train_apps_by_name[args.app](dtype, n=args.batch)
        if args.layer is not None: app = U.apps.Trace([app.oplist[args.layer]])
        app.train()

    logging.debug(f'Arch: {arch}')
    logging.debug(f'App: {app}')
    logging.debug(f'Arch Peak ops/cyc/core: {arch.peak_opc(dtype)}')
    logging.debug(f'App Flops: {app.flops}')

    sim_args = dict(
        placement_mode=args.placement_mode,
        l1_capacity=args.l1_capacity,
        l1_assoc=args.l1_assoc)

    if args.parallel: time_ns, layers = simulate_app_par(arch, app, sim_args)
    else: time_ns, layers = simulate_app(arch, app, sim_args)

    cycles = sum(result.cycles for result in layers)
    logging.info('App Summary:')
    logging.info(f'+ Simulation time: {time_ns / 1e9} s')
    logging.info(f'+ Total Latency: {cycles} cyc')
    logging.info(f'+ Throughput: {arch.freq / cycles * args.batch} samp/sec')
    logging.info(f'+ Compute: {app.flops / cycles} flops/cyc')
    logging.info(f'+ Compute: {app.flops / cycles / arch.ntiles} flops/cyc/core')
    logging.info(f'+ Efficiency: {app.flops / cycles / arch.ntiles / arch.peak_opc(dtype) * 100} %')
