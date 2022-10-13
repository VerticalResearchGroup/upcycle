import sys
import torch
import upcycle as U

dev = torch.device('cuda:0')

app, trace, batch, dtype = U.apps.workload_factory('resnet50', 'offline', infer=True)
a100_peak_fp16 = 312e12

total_lat = 0
for op in trace.oplist:
    avg_time = U.torchrun.time_torch_op(op, dev)
    op_util_a100 = op.flops / avg_time / a100_peak_fp16
    total_lat += avg_time
    print(f'{op}:  \t\tavg. time={avg_time:.5f}\tutil={op_util_a100 * 100:.2f}%')

print(f'Total latency: {total_lat:.5f} sec')
print(f'Throughput: {trace.bs / total_lat:.2f} images/sec')
