from .. import apps

def analyze_trace(trace : apps.Trace):
    llc_cap = 0
    for op in trace:
        print(op)
        llc_cap = max(llc_cap, op.min_llc_capacity)

    print(f'LLC Capacity: {llc_cap / 2**20}')


app : apps.App
for (name, app) in apps.mlperf_v1_apps.items():
    if app.bs.infer_online is not None: analyze_trace(app.default_infer_online())
    if app.bs.infer_offline is not None: analyze_trace(app.default_infer_offline())
    if app.bs.train_small is not None: analyze_trace(app.default_train_small())
    if app.bs.train_large is not None: analyze_trace(app.default_train_large())
