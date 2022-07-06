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

if __name__ == '__main__':
    counter = None
    lock = None

def init_pool_processes(c, l):
    global counter
    global lock

    counter = c
    lock = l

def simulate_layer(arch : U.Arch, op : U.ops.Operator, sim_kwargs):
    logger.debug(f'Simulating {op}...')
    global counter
    global lock
    sim_kwargs['counter'] = counter
    sim_kwargs['lock'] = lock
    result = U.model.simulate(arch, op, **sim_kwargs)
    logger.debug(f'Finished {op}')
    return result

def log_layer(arch : U.Arch, dtype : U.Dtype, app : U.apps.Trace, i, op : U.ops.Operator, result : U.model.SimResult, time_ns=None, details=True):

    gops = int(op.flops / result.cycles * arch.freq / 1e9)
    eff = np.round(op.flops / result.cycles / arch.ntiles / arch.peak_opc(dtype) * 100, 2)
    logger.info(f'{green}Layer {i}/{len(app.oplist)}: {op} {reset}')
    if details:
        if time_ns is not None: logger.info(f'+ Simulation time: {time_ns / 1e9} s')
        logger.info(f'+ Latency: {int(result.cycles)} cyc, Hops: {np.sum(result.traffic)}')
        logger.info(f'+ Compute: {gops} Gops ({blue}Efficiency: {eff} %{reset})')
        if 'avg_dests' in result.kwstats:
            logger.info(f'+ (Max) Avg Dests Per Line: {np.max(result.kwstats["avg_dests"])}')
        if 'avg_groups' in result.kwstats:
            logger.info(f'+ (Max) Avg Groups Per Line: {np.max(result.kwstats["avg_groups"])}')
        if 'max_lines' in result.kwstats:
            logger.info(f'+ (Max) lines transmitted in one step : {np.max(result.kwstats["max_lines"])}')


def simulate_app(arch : U.Arch, dtype : U.Dtype, app : U.apps.Trace, sim_args, verbose=True):
    layers = []
    cache = {}
    total_steps = sum(U.model.num_steps(arch, op, **sim_args) for op in app.oplist)
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
        if verbose: log_layer(arch, dtype, app, i, op, result, t1 - t0, details=not hit)

    tt1 = time.perf_counter_ns()
    return tt1 - tt0, layers

def simulate_app_par(parallel : int, arch : U.Arch, dtype : U.Dtype, app : U.apps.Trace, sim_args, verbose=True):
    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()

    pool = multiprocessing.Pool(
        parallel, initializer=init_pool_processes, initargs=(counter, lock))
    tt0 = time.perf_counter_ns()
    unique_ops = list(app.unique_ops)
    total_steps = sum(U.model.num_steps(arch, op, **sim_args) for op in unique_ops)

    progress = tqdm.tqdm(total=total_steps, unit='steps', smoothing=0.05)

    result = pool.map_async(
        functools.partial(simulate_layer, arch, sim_kwargs=sim_args), unique_ops)

    last = 0
    while not result.ready():
        result.wait(0.5)
        cur = counter.value
        progress.update(cur - last)
        last = cur

    progress.close()
    unique_results = result.get()

    cache = {op: result for op, result in zip(unique_ops, unique_results)}
    tt1 = time.perf_counter_ns()

    layers = []
    for i, op in enumerate(app.oplist):
        result = cache[op]
        layers.append(result)
        if verbose: log_layer(arch, dtype, app, i, op, result)

    return tt1 - tt0, layers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an application')
    parser.add_argument('-r', '--arch', type=str, default='oracle')
    parser.add_argument('-d', '--dtype', type=str, default='')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-T', '--bwd-only', action='store_true')
    parser.add_argument('-i', '--infer', action='store_true')
    parser.add_argument('-a', '--app', type=str, default='resnet50')
    parser.add_argument('-b', '--batch', type=str, default='1')
    parser.add_argument('-m', '--placement-mode', type=str, default='pg')
    parser.add_argument('-p', '--parallel', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('-l', '--layer', type=int, default=None)

    parser.add_argument('--noc-ports', type=int, default=1)
    parser.add_argument('--l1-capacity', type=int, default=64*1024)
    parser.add_argument('--l1-assoc', type=int, default=16)
    parser.add_argument('--line-size', type=int, default=64)
    parser.add_argument('--bgsize', type=str, default='4,8')

    args = parser.parse_args()
    assert not (args.train and args.infer)
    assert args.train or args.infer

    if args.debug:
        assert args.parallel == 1, f'Cannot debug in parallel'
        logger.setLevel(logging.DEBUG)

    if args.arch == 'oracle':
        arch = U.OracleArch(
            2.4e9, 512, 1, 32, 64,
            args.noc_ports, args.line_size, args.l1_capacity, args.l1_assoc)
    elif args.arch == 'bg':
        [grows, gcols] = list(map(int, args.bgsize.split(',')))
        arch = U.BgroupArch(
            2.4e9, 512, 1, 32, 64,
            args.noc_ports, args.line_size, args.l1_capacity, args.l1_assoc,
            grows, gcols)


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
    logging.info(f'Arch: {arch}')

    sim_args = dict(placement_mode=args.placement_mode)

    if args.parallel > 1: time_ns, layers = simulate_app_par(args.parallel, arch, dtype, app, sim_args, args.verbose)
    else: time_ns, layers = simulate_app(arch, dtype, app, sim_args, args.verbose)

    cycles = sum(result.cycles for result in layers)
    logging.info(f'Summary: (Simulation time: {time_ns / 1e9} s)')
    logging.debug(f'+ Total Latency: {cycles} cyc')
    logging.info(f'+ Throughput: {green}{arch.freq / cycles * batch} samp/sec{reset}, {blue}{app.flops / cycles / arch.ntiles / arch.peak_opc(dtype) * 100} %{reset}')
    logging.debug(f'+ Compute: {app.flops / cycles} flops/cyc')
    logging.debug(f'+ Compute: {app.flops / cycles / arch.ntiles} flops/cyc/core')

    if args.verbose:
        max_lines = max(result.kwstats.get('max_lines', -1) for result in layers)
        if max_lines > 0:
            logger.info(f'+ Max Lines transmitted in one step: {max_lines}')
        tot_lines = max(result.kwstats.get('tot_lines', -1) for result in layers)
        if tot_lines > 0:
            logger.info(f'+ Total Lines transmitted: {tot_lines} ({tot_lines * 2 / 2**20} MB)')
