import sys
import time
import os
import upcycle as U
import numpy as np
from matplotlib import pyplot as plt

B = 128
app, trace, batch, dtype = U.apps.workload_factory('resnet50', B, infer=True, layer=0)
print(f'Total flops = {trace.flops}')
print(f'Per sample flops = {trace.flops / batch}')

N = 10000
times, utils = list(U.torchrun.run_with_torch(trace, device_type='cuda', device_id=3, niters=N, warmup=1000))[0]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

smoothed = moving_average(utils, 20)

plt.plot(np.arange(len(smoothed)), smoothed)
plt.plot([0, len(smoothed)], [1.0, 1.0], 'r--')
plt.tight_layout()
os.makedirs('./figs', exist_ok=True)
plt.savefig('figs/times.pdf')
