import upcycle as U
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

arch = U.OracleArch(2e9, 512, 1, 32, 64)
logging.info(f'Arch: {arch}')
logging.debug(f'Arch Peak ops/cyc/core: {arch.peak_opc(U.Dtype.I8)}')
app = U.apps.testconv(U.Dtype.I8, n=1)
app.optimize_linears_for_infer()
soc = U.model.make_soc(arch, l1_capacity=32*1024, l1_assoc=4)

logging.info(f'Op: {app.oplist[0]}')
logging.debug(f'Flops: {app.oplist[0].flops}')

soc.simulate(app.oplist[0])

logging.info(f'Total Latency: {soc.cycles} cyc')
logging.info(f'Compute: {app.flops / soc.cycles} flops/cyc')
logging.info(f'Compute: {app.flops / soc.cycles / arch.ntiles} flops/cyc/core')
logging.info(f'Efficiency: {app.flops / soc.cycles / arch.ntiles / arch.peak_opc(U.Dtype.I8) * 100} %')

# fig, ax = plt.subplots(figsize=(16, 8))
# ax.set_xlim([0, 64])
# ax.set_ylim([0, 32])

# anim = soc.animate(fig, ax)
# plt.tight_layout()
# # plt.show()
# anim.save('anim.gif')
