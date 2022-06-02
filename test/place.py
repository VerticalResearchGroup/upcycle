import upcycle as U

arch = U.Arch(32, 64)
app = U.apps.testapp(U.Dtype.I8, n=512)
app.optimize_linears_for_infer()
print(f'App has {app.flops} flops')

gwl = U.model.place_op('naive', arch, app.oplist[0])


wi = gwl.tiles[0][0]
for t in wi.read_trace:
    for l in t.lines:
        print(hex(l))