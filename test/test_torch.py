import sys
import upcycle as U

app, trace, batch, dtype = U.apps.workload_factory('resnet50', 'offline', infer=True)

N = int(sys.argv[1])
results = U.torchrun.run_with_torch(trace, device_type='cuda', device_id=0, niters=N)

a100_peak_fp16 = 312e12

total_lat = 0
for op, lat_sec in zip(trace.oplist, results):
    op_util_a100 = op.flops / lat_sec / a100_peak_fp16
    total_lat += lat_sec
    print(f'{op}:  \t\tavg. time={lat_sec:.5f}\tutil={op_util_a100 * 100:.2f}%')


print(f'Total latency: {total_lat:.5f} sec')
print(f'Throughput: {trace.bs / total_lat:.2f} images/sec')
