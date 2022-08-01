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

def simulate_layer(key, sim_kwargs):
    arch, op = key
    logger.debug(f'Simulating {op}...')
    global counter
    global lock
    sim_kwargs['counter'] = counter
    sim_kwargs['lock'] = lock
    result = U.model.simulate(arch, op, **sim_kwargs)
    logger.debug(f'Finished {op}')
    return result

def simulate_apps_par(parallel : int, workloads : list[tuple[U.Arch, U.apps.Trace]], sim_args):
    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()

    pool = multiprocessing.Pool(
        parallel, initializer=init_pool_processes, initargs=(counter, lock))

    tt0 = time.perf_counter_ns()
    unique_ops = set()
    for arch, trace in workloads:
        for op in trace.unique_ops:
            unique_ops.add((arch, op))

    logger.info(f'Computing number of steps for {len(unique_ops)} unique ops...')
    total_steps = sum(U.model.num_steps(arch, op, **sim_args) for arch, op in unique_ops)

    logger.info(f'Running Simulation for {len(unique_ops)} unique ops...')
    progress = tqdm.tqdm(total=total_steps, unit='steps', smoothing=0.05)

    result = pool.map_async(
        functools.partial(simulate_layer, sim_kwargs=sim_args), unique_ops)

    last = 0
    while not result.ready():
        result.wait(0.5)
        cur = counter.value
        progress.update(cur - last)
        last = cur

    progress.close()
    unique_results = result.get()

    cache = {key: result for key, result in zip(unique_ops, unique_results)}
    tt1 = time.perf_counter_ns()

    return tt1 - tt0, cache

def make_arch(arch_name, noc_ports):
    arch_kwargs = dict(
        noc_ports=noc_ports,
        line_size=32,
        l1_capacity=64 * 1024,
        l1_assoc=8,
        group='4,8')
    return U.arch.arch_factory(arch_name, **arch_kwargs)

def make_workload(args):
    app, scenario, arch_name, nocports = args
    arch = make_arch(arch_name, nocports)
    app, trace, _, _ = U.apps.workload_factory(app, scenario, infer=True)
    return arch, trace

def print_results(args, app_name, scenario, arch, noc_ports, results):
    arch = make_arch(arch, noc_ports)
    app, trace, batch, dtype = U.apps.workload_factory(app_name, scenario, infer=True)

    logging.info('===========================')
    logging.info(f'Workload: {app_name}, Inference, {dtype}, bs={batch}, {trace.flops / 1e9} GOPS')
    logging.info(f'Arch: {arch}')

    layers = [
        results[(arch, op)]
        for op in trace.unique_ops
    ]

    cycles = sum(result.cycles for result in layers)
    throughput = arch.freq / cycles * batch
    util = trace.flops / cycles / arch.ntiles / arch.peak_opc(dtype)
    logging.info(f'+ Throughput: {green}{throughput} samp/sec{reset}, {blue}{util * 100} %{reset}')
    logging.info('===========================')


infer_apps = [
    (app, scenario, arch_name, nocports)
    for app in ['resnet50', 'bert-large-squad', 'ssdrn34-1200', 'rnnt']
    for arch_name in ['oracle', 'bg']
    for nocports in [1, 2]
    for scenario in ['online', 'offline']
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate All Inference Apps and Print Results')
    parser.add_argument('-p', '--parallel', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    workloads = list(map(make_workload, infer_apps))
    sim_args = dict(placement_mode='pg')
    time_ns, results = simulate_apps_par(args.parallel, workloads, sim_args)

    for app, scenario, arch_name, noc_ports in infer_apps:
        print_results(args, app, scenario, arch_name, noc_ports, results)



