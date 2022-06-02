import upcycle as U


print('=== Forward Only ===')
bl = U.apps.bertlarge(U.Dtype.I8, n=1, s=512)
print('Bert Large 512: ', bl.flops / 1e9)

bb = U.apps.bertbase(U.Dtype.I8, n=1, s=512)
print('Bert Base 512: ', bb.flops / 1e9)

rn50 = U.apps.resnet50(U.Dtype.I8, n=1)
print('Resnet50: ', rn50.flops / 1e9)

ssd1200 = U.apps.ssdrn34_1200(U.Dtype.I8, n=1)
print('SSD-Resnet34 (1200x1200): ', ssd1200.flops / 1e9)

ssd300 = U.apps.ssdrn34_300(U.Dtype.I8, n=1)
print('SSD-Resnet34 (300x300): ', ssd300.flops / 1e9)

rnnt_infer = U.apps.rnnt_infer(U.Dtype.I8, n=1, il=389, ol=195)
print('RNN-T (389 -> 195): ', rnnt_infer.flops)

print()
print('=== Forward + Backward ===')
bl.train()
print('Bert Large 512: ', bl.flops / 1e9)

bb.train()
print('Bert Base 512: ', bb.flops / 1e9)
