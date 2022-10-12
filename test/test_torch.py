import upcycle as U

app, trace, batch, dtype = U.apps.workload_factory('resnet50', 'offline', infer=True)

results = U.torchrun.run_with_torch(trace, device_type='cuda', device_id=0, niters=100)

a100_peak_fp16 = 312e12

for op, lat_sec in zip(trace.oplist, results):
    op_util_a100 = op.flops / lat_sec / a100_peak_fp16
    print(f'{op}:  \t\tavg. time={lat_sec:.5f}\tutil={op_util_a100 * 100:.2f}%')


