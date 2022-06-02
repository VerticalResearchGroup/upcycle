import upcycle as U
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

arch = U.FlatMeshArch(2e9, 512, 1, 32, 64)
app = U.apps.testapp(U.Dtype.I8, n=512)
soc = U.model.make_soc(arch, randomize_llc=True)

soc.simulate(app.oplist[0])

print(f'Total Latency: {soc.cycles} cyc')
print(f'Compute: {app.flops / soc.cycles} flops/cyc')
print(f'Compute: {app.flops / soc.cycles / arch.ntiles} flops/cyc/core')
print(f'Efficiency: {app.flops / soc.cycles / arch.peak_opc(U.Dtype.I8) * 100} %')
