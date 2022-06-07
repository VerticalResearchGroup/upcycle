import upcycle as U

teng = U.pat.Component(7, 0, 1/2e9, 0)
teng.scale_isopower(32)

print(teng)
print(int(1 / teng.delay) / 1e6)
