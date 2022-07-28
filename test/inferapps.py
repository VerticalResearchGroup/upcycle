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

    return tt1 - tt0, layers

def process_app(args, app_name, scenario, arch, noc_ports, placement_mode='pg'):
    arch_kwargs = dict(
        noc_ports=noc_ports,
        line_size=32,
        l1_capacity=64 * 1024,
        l1_assoc=8)

    arch = U.arch.arch_factory(arch, **arch_kwargs)
    app, trace, batch, dtype = U.apps.workload_factory(app_name, scenario, infer=True)

    logging.info(f'Workload: {app_name}, Inference, {dtype}, bs={batch}, {trace.flops / 1e9} GOPS')
    logging.info(f'Arch: {arch}')
    sim_args = dict(placement_mode=placement_mode)
    time_ns, layers = simulate_app_par(args.parallel, arch, dtype, trace, sim_args, args.verbose)
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

    for app, scenario, arch_name, noc_ports in infer_apps:
        process_app(args, app, scenario, arch_name, noc_ports)



