import upcycle as U

arch =  U.arch.arch_factory('hier', dict(
    vbits=512,
    geom=U.arch.ntiles_to_geom(2048),
    compute_scale=[0.5, 2, 10, 100, 0],
    noc_scale=[0.5, 2, 10, 100, 0]))

arch_ext = U.simdb.ArchExtConfig(2.4e9, CU.base_membw, 2.0, 2.0, 1.0)
rn50 = U.apps.mlperf_v1_apps['resnet50'].default_infer_online()
db = U.simdb.SimDb(arch)

print(f'area: {db.area_mm2():.2f} mm2')

trace = db.trace(rn50, arch_ext)
for ld in trace:
    print(ld.op)
    print(f'+    cycles: {ld.real_cyc} ({ld.onchip_cycles}, {ld.mem_cyc})')
    print(f'+    power: {ld.power_w}')
    print()

print(f'Max power: {max(ld.power_w for ld in trace)} W')
print(f'Total cycles: {sum(ld.real_cyc for ld in trace)}')
