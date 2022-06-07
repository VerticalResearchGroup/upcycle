import upcycle as U
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import logging
import time
import multiprocessing

logging.basicConfig(level=logging.INFO)
arch = U.OracleArch(2e9, 512, 1, 32, 64)
app = U.apps.resnet50(U.Dtype.I8, n=1)
# app = U.apps.Trace([
#     U.ops.Conv2D(dtype=U.Dtype.I8, train=True, n=1, h=28, w=28, c=128, p=28, q=28, k=128, r=3, s=3, stride=1)
# ])
app.optimize_linears_for_infer()

def simulate_layer(arch, op : U.ops.Operator):
    soc.simulate(op)
    return soc

blue = '\x1b[38;5;39m'
reset = '\x1b[0m'

if __name__ == '__main__':
    # pool = multiprocessing.Pool(24)
    logging.info(f'Arch: {arch}')
    logging.debug(f'Arch Peak ops/cyc/core: {arch.peak_opc(U.Dtype.I8)}')
    logging.debug(f'App Flops: {app.flops}')

    tt0 = time.perf_counter_ns()
    for i, op in enumerate(app.oplist):
        logging.info(f'Layer {i}/{len(app.oplist)}: {op}')

        t0 = time.perf_counter_ns()
        soc = U.model.make_soc(arch, l1_capacity=32*1024, l1_assoc=4)
        soc.simulate(op)
        t1 = time.perf_counter_ns()

        logging.info(f'+ Simulation time: {(t1 - t0) / 1e9} s')
        logging.info(f'+ Total Latency: {soc.cycles} cyc')
        logging.info(f'+ Compute: {op.flops / soc.cycles} flops/cyc')
        logging.info(f'+ Compute: {op.flops / soc.cycles / arch.ntiles} flops/cyc/core')
        logging.info(f'{blue}+ Efficiency: {op.flops / soc.cycles / arch.ntiles / arch.peak_opc(U.Dtype.I8) * 100} %{reset}')

    tt1 = time.perf_counter_ns()
    logging.info('App Summary:')
    logging.info(f'+ Simulation time: {(tt1 - tt0) / 1e9} s')
    # logging.info(f'+ Total Latency: {cycles} cyc')
    # logging.info(f'+ Compute: {app.flops / cycles} flops/cyc')
    # logging.info(f'+ Compute: {app.flops / cycles / arch.ntiles} flops/cyc/core')
    # logging.info(f'+ Efficiency: {app.flops / cycles / arch.ntiles / arch.peak_opc(U.Dtype.I8) * 100} %')
