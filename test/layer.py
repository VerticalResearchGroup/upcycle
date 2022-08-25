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

if __name__ == '__main__':
    counter = None
    lock = None

def init_pool_processes(c, l):
    global counter
    global lock

    counter = c
    lock = l

def simulate_layer(arch : U.Arch, op : U.ops.Operator, sim_kwargs):
    # tracemalloc.start()
    logger.debug(f'Simulating {op}...')
    global counter
    global lock
    sim_kwargs['counter'] = counter
    sim_kwargs['lock'] = lock
    result = U.model.simulate(arch, op, **sim_kwargs)
    logger.debug(f'Finished {op}')
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # logger.debug("[ Top 10 memory allocation sites ]")
    # for stat in top_stats[:10]:
    #     logger.debug(stat)
    # logger.debug("[ ============================== ]")

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


def simulate_app(arch : U.Arch, dtype : U.Dtype, app : U.apps.Trace, verbose=True):
    layers = []
    cache = {}
    total_steps = sum(U.model.num_steps(arch, op) for op in app.oplist)
    tt0 = time.perf_counter_ns()
    for i, op in enumerate(app.oplist):
        t0 = time.perf_counter_ns()
        if op not in cache:
            result = simulate_layer(arch, op, {})
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

def get_arch(args):
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

        return arch

def get_workload(args):
    app = U.apps.mlperf_v1_apps[args.app]

    if args.infer:
        if args.batch in {'offline', 'online'}:
            batch = app.bs.infer_offline if args.batch == 'offline' \
                else app.bs.infer_online
        else: batch = int(args.batch)

        if args.dtype == '': dtype = app.infer_dtype
        else: dtype = U.Dtype.from_str(args.dtype)

        trace = app.infer_factory(dtype, n=batch)
        if args.layer is not None: app = U.apps.Trace([trace.oplist[args.layer]])
        trace.infer()

    else:
        if args.batch in {'large', 'small'}:
            batch = app.bs.train_large if args.batch == 'large' \
                else app.bs.train_small
        else: batch = int(args.batch)

        if args.dtype == '': dtype = app.train_dtype
        else: dtype = U.Dtype.from_str(args.dtype)

        trace = app.train_factory(dtype, n=batch)
        if args.layer is not None: app = U.apps.Trace([trace.oplist[args.layer]])
        trace.train(args.bwd_only)

    return app, trace, batch, dtype

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an application')

    U.arch.arch_cli_params(parser)
    U.apps.workload_cli_params(parser)

    parser.add_argument('-p', '--parallel', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    logger.setLevel(logging.DEBUG)

    assert args.layer is not None

    arch = U.arch.arch_from_cli(args)
    app, trace, batch, dtype = U.apps.workload_from_cli(args)

    logging.info(f'App: {args.app} ({"train" if args.train else "infer"})')
    logging.info(f'Dtype: {dtype}, Batch Size: {batch}')
    logging.info(f'App Ops: {trace.flops / 1e9} G')
    logging.info(f'Arch: {arch}')

    time_ns, layers = simulate_app(arch, dtype, trace, args.verbose)
    cycles = sum(result.cycles for result in layers)

    logging.info(f'Summary: (Simulation time: {time_ns / 1e9} s)')
    logging.debug(f'+ Total Latency: {cycles} cyc')
    logging.info(f'+ Throughput: {green}{arch.freq / cycles * batch} samp/sec{reset}, {blue}{trace.flops / cycles / arch.ntiles / arch.peak_opc(dtype) * 100} %{reset}')
    logging.debug(f'+ Compute: {trace.flops / cycles} flops/cyc')
    logging.debug(f'+ Compute: {trace.flops / cycles / arch.ntiles} flops/cyc/core')

    if args.verbose:
        max_lines = max(result.kwstats.get('max_lines', -1) for result in layers)
        if max_lines > 0:
            logger.info(f'+ Max Lines transmitted in one step: {max_lines}')
        tot_lines = max(result.kwstats.get('tot_lines', -1) for result in layers)
        if tot_lines > 0:
            logger.info(f'+ Total Lines transmitted: {tot_lines} ({tot_lines * 2 / 2**20} MB)')
