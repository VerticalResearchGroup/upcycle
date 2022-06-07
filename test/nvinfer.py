import upcycle as U
import numpy as np

for name, stats in U.nvdb.a100_perf.items():
    peak = U.nvdb.a100_peak[stats.infer_dtype]
    app_flops = U.apps.app_infer_flops[name]

    on_perf = stats.infer_online_perf
    on_eff = np.round(on_perf * app_flops / peak * 100, 2)

    off_perf = stats.infer_offline_perf
    off_eff = np.round(off_perf * app_flops / peak * 100, 2)

    print(f'{name}: {app_flops} flops, online: {on_eff}%, offline: {off_eff}%')
