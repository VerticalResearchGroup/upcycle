import upcycle as U
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

arch = U.OracleArch(2e9, 512, 1, 32, 64)
app = U.apps.testapp(U.Dtype.I8, n=512)
app.optimize_linears_for_infer()
soc = U.model.make_soc(arch, randomize_llc=False)

soc.simulate(app.oplist[0])

print(f'Total Latency: {soc.cycles} cyc')
print(f'Compute: {app.flops / soc.cycles} flops/cyc')
print(f'Compute: {app.flops / soc.cycles / arch.ntiles} flops/cyc/core')
print(f'Efficiency: {app.flops / soc.cycles / arch.peak_opc(U.Dtype.I8) * 100} %')

# fig, ax = plt.subplots(figsize=(16, 8))
# ax.set_xlim([0, 64])
# ax.set_ylim([0, 32])

# anim = soc.animate(fig, ax)
# plt.tight_layout()
# plt.show()
