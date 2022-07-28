import upcycle as U

arch = U.OracleArch(
    2.4e9, 512, 1, 32, 64,
    1, 32, 64 * 1024, 16)

t = U.model.Tensor(arch, 1, U.Dtype.I8, (3, 224, 224))

addrs = t[:, 0:8, 0]
print([hex(a) for a in addrs])

U.model.common.USE_C_TRACE = True

tile = t[:, 0:8, 0][0]
# print(tile)

l1 = U.model.c_model.Cache(arch.l1_nset, arch.l1_assoc, arch.lbits)
dl = U.model.c_model.DestList()

U.model.c_model.tile_read_trace(l1, dl, tile, 0)

