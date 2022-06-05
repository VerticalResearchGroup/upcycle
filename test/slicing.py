import upcycle as U


x = U.model.Tensor(1, U.model.Dtype.I8, (1, 224, 224, 3))

for line in x[0, 0:2, 0:44, :]:
    print(hex(line))
