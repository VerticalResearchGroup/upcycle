import upcycle as U

app, trace, batch, dtype = U.apps.workload_factory('resnet50', 'offline', infer=True, layer=0)

results = U.torchrun.run_with_torch(trace, device_type='cuda', device_id=0, niters=1000)

for op, lat_sec in zip(trace.oplist, results):
    print(f'{op} : {result}')


