import upcycle as U
import numpy as np
import pandas as pd

def train_data(name, stats : U.nvdb.NvidiaAppStats):
    peak = U.nvdb.a100_peak[stats.train_dtype]
    app_flops = U.apps.mlperf_v1_apps[name].train_flops

    large_perf = stats.train_large_perf
    large_eff = np.round(large_perf * app_flops / peak * 100, 2)

    if stats.train_small_perf is not None:
        small_perf = stats.train_small_perf
        small_eff = np.round(small_perf * app_flops / peak * 100, 2)
    else:
        small_perf = None
        small_eff = None

    return name, np.round(app_flops / 1e9, 2), large_perf, large_eff, small_perf, small_eff


def infer_data(name, stats : U.nvdb.NvidiaAppStats):
    peak = U.nvdb.a100_peak[stats.infer_dtype]
    app_flops = U.apps.mlperf_v1_apps[name].infer_flops

    on_perf = stats.infer_online_perf
    on_eff = np.round(on_perf * app_flops / peak * 100, 2)

    off_perf = stats.infer_offline_perf
    off_eff = np.round(off_perf * app_flops / peak * 100, 2)

    return name, np.round(app_flops / 1e9, 2), on_perf, on_eff, off_perf, off_eff

idata = {
    infer_data(name, stats)
    for name, stats in U.nvdb.a100_perf.items()
    if stats.infer_online_perf is not None
}

infer_dframe = pd.DataFrame({
    'Name': [d[0] for d in idata],
    'GOPs': [d[1] for d in idata],
    'On. Perf.': [d[2] for d in idata],
    'On. Eff. (%)': [d[3] for d in idata],
    'Off. Perf.': [d[4] for d in idata],
    'Off. Eff. (%)': [d[5] for d in idata]
})

tdata = {
    train_data(name, stats)
    for name, stats in U.nvdb.a100_perf.items()
    if stats.train_large_perf is not None
}

train_dframe = pd.DataFrame({
    'Name': [d[0] for d in tdata],
    'GOPs': [d[1] for d in tdata],
    'Small Perf.': [d[4] for d in tdata],
    'Small Eff. (%)': [d[5] for d in tdata],
    'Large Perf.': [d[2] for d in tdata],
    'Large Eff. (%)': [d[3] for d in tdata],
})

print('Inference Data:')
print(infer_dframe)
print()
print('Training Data:')
print(train_dframe)
print()

# print(infer_dframe.style.to_latex())
