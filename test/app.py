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
ch.setFormatter(U.logutils.CustomFormatter())
logging.basicConfig(level=logging.INFO, handlers=[ch])

def simulate_layer(op : U.ops.Operator, soc_args):
    soc = U.model.make_soc(arch, **soc_args)
    logger.debug(f'Simulating {op}...')
    soc.simulate(op)
    logger.debug(f'Finished {op}')
    return soc

def log_layer(arch : U.Arch, app : U.apps.Trace, i, op : U.ops.Operator, soc, time_ns=None):
    blue = '\x1b[38;5;39m'
    green = '\033[92m'
    reset = '\x1b[0m'
    gops = int(op.flops / soc.cycles * arch.freq / 1e9)
    eff = np.round(op.flops / soc.cycles / arch.ntiles / arch.peak_opc(U.Dtype.I8) * 100, 2)
    logger.info(f'{green}Layer {i}/{len(app.oplist)}: {op} {reset}')
    if time_ns is not None: logger.info(f'+ Simulation time: {time_ns / 1e9} s')
    logger.info(f'+ Latency: {int(soc.cycles)} cyc, Hops: {soc.total_hops}')
    logger.info(f'+ Compute: {gops} Gops ({blue}Efficiency: {eff} %{reset})')


def simulate_app(app : U.apps.Trace, soc_args):
    socs = []
    cache = {}
    tt0 = time.perf_counter_ns()
    for i, op in enumerate(app.oplist):
        t0 = time.perf_counter_ns()
        if op not in cache:
            soc = simulate_layer(op, soc_args)
            cache[op] = soc
        else:
            soc = cache[op]

        socs.append(soc)

        t1 = time.perf_counter_ns()
        log_layer(arch, app, i, op, soc, t1 - t0)

    tt1 = time.perf_counter_ns()
    return tt1 - tt0, socs

def simulate_app_par(app : U.apps.Trace, soc_args):
    pool = multiprocessing.Pool(16)
    tt0 = time.perf_counter_ns()
    unique_ops = list(app.unique_ops)
    unique_socs = pool.map(
        functools.partial(simulate_layer, soc_args=soc_args), unique_ops)

    cache = {op: soc for op, soc in zip(unique_ops, unique_socs)}
    tt1 = time.perf_counter_ns()

    socs = []
    for i, op in enumerate(app.oplist):
        soc = cache[op]
        socs.append(soc)
        log_layer(arch, app, i, op, soc)

    return tt1 - tt0, socs


if __name__ == '__main__':
    arch = U.OracleArch(2e9, 512, 1, 32, 64, 1)
    # app = U.apps.bertbase(U.Dtype.I8, n=64)
    app = U.apps.resnet50(U.Dtype.I8, n=16)
    # app = U.apps.rnnt_infer(U.Dtype.FP16, n=64, il=16)
    # app = U.apps.Trace([app.oplist[0]])
    app.infer()


    logging.info(f'Arch: {arch}')
    logging.debug(f'Arch Peak ops/cyc/core: {arch.peak_opc(U.Dtype.I8)}')
    logging.debug(f'App Flops: {app.flops}')

    soc_args = dict(placement_mode='flatmap', l1_capacity=64*1024, l1_assoc=16)

    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        time_ns, socs = simulate_app_par(app, soc_args)
    else:
        time_ns, socs = simulate_app(app, soc_args)

    cycles = sum(soc.cycles for soc in socs)
    logging.info('App Summary:')
    logging.info(f'+ Simulation time: {time_ns / 1e9} s')
    logging.info(f'+ Total Latency: {cycles} cyc')
    logging.info(f'+ Compute: {app.flops / cycles} flops/cyc')
    logging.info(f'+ Compute: {app.flops / cycles / arch.ntiles} flops/cyc/core')
    logging.info(f'+ Efficiency: {app.flops / cycles / arch.ntiles / arch.peak_opc(U.Dtype.I8) * 100} %')
