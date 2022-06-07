import upcycle as U
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import logging
import multiprocessing
import time

logging.basicConfig(level=logging.INFO)
arch = U.OracleArch(2e9, 512, 1, 32, 64)
app = U.apps.resnet50(U.Dtype.I8, n=1)
app.optimize_linears_for_infer()

def simulate_layer(op : U.ops.Operator):
    soc = U.model.make_soc(arch, l1_capacity=32*1024, l1_assoc=4)
    logging.debug(f'Simulating {op}...')
    soc.simulate(op)
    logging.info(f'Finished {op}')
    return soc


if __name__ == '__main__':
    pool = multiprocessing.Pool(24)
    logging.info(f'Arch: {arch}')
    logging.debug(f'Arch Peak ops/cyc/core: {arch.peak_opc(U.Dtype.I8)}')
    logging.debug(f'App Flops: {app.flops}')

    tt0 = time.perf_counter_ns()
    socs = pool.map(simulate_layer, app.oplist)
    tt1 = time.perf_counter_ns()

    for i, (op, soc) in enumerate(zip(app.oplist, socs)):
        logging.info(f'Layer {i}/{len(app.oplist)}: {op}')
        logging.info(f'+ Total Latency: {soc.cycles} cyc')
        logging.info(f'+ Total Hops: {soc.total_hops}')
        logging.info(f'+ Compute: {op.flops / soc.cycles} flops/cyc')
        logging.info(f'+ Compute: {op.flops / soc.cycles / arch.ntiles} flops/cyc/core')
        logging.info(f'+ Efficiency: {op.flops / soc.cycles / arch.ntiles / arch.peak_opc(U.Dtype.I8) * 100} %')

    cycles = sum(soc.cycles for soc in socs)
    logging.info('App Summary:')
    logging.info(f'+ Simulation time: {(tt1 - tt0) / 1e9} s')
    logging.info(f'+ Total Latency: {cycles} cyc')
    logging.info(f'+ Compute: {app.flops / cycles} flops/cyc')
    logging.info(f'+ Compute: {app.flops / cycles / arch.ntiles} flops/cyc/core')
    logging.info(f'+ Efficiency: {app.flops / cycles / arch.ntiles / arch.peak_opc(U.Dtype.I8) * 100} %')
