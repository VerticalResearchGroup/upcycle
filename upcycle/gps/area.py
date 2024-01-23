
from .. import pat
import numpy as np

def area_scale(from_node, to_node):
    factors = {
        16: 1.0,
        12: 0.86,
        7: 0.34,
        5: 0.19,
        3: 0.19/1.5,
    }

    return factors[from_node] / factors[to_node]

def largest_pow_2_leq(x):
    return 2 ** int(np.log2(x))


def galileo_area(num_tiles, simd_width, node):
    int8_mac = pat.Component(32, 336.672e-6, 1 / 1e9, 283e-6).scale_isofreq(12)
    vrf_1k_proto = pat.Component(20, 20000e-12, 1 / 1e9, 1.0).scale_isofreq(12)

    vrf_1k = pat.Component(
        7,
        vrf_1k_proto.area,
        vrf_1k_proto.delay,
        vrf_1k_proto.power * 1.3)

    core = pat.Component(45, 0.044, 1 / 1e9, 14e-3).scale_isofreq(12)

    pe_core_area = \
        (int8_mac.area * 3 * (simd_width // 8)) +  vrf_1k.area * 2 + core.area

    l1_cacti = pat.cacti.cacti(32 * 1024, node=12)
    l2_cacti = pat.cacti.cacti(16 * 1024, node=12)
    llc_cacti = pat.cacti.cacti(max(32 * 2**20 / largest_pow_2_leq(num_tiles), 4096), node=12)


    l1_area = l1_cacti.area_mm2
    l2_slice_area = l2_cacti.area_mm2
    llc_slice_area = llc_cacti.area_mm2
    pe_area = pe_core_area + l1_area + l2_slice_area + llc_slice_area

    fp16_peak = num_tiles * simd_width / 8 * 2 * 2.4e9 / 1e12

    print(f'==== {num_tiles} core, {simd_width}-bit Galileo @ {node}nm ====')
    print(f'+ Total FP16 Compute: {fp16_peak}')
    # print(f'+ PE area: {pe_area:.4f} mm^2')
    # print(f'+     Core: {pe_core_area:.4f} mm^2')
    # print(f'+     L1: {l1_area:.4f} mm^2')
    # print(f'+     L2 slice: {l2_slice_area:.4f} mm^2')
    # print(f'+     LLC slice: {llc_slice_area:.4f} mm^2')
    print(f'Total area: {pe_area * num_tiles / area_scale(12, node):.4f} mm^2')
    print()

print(f'12nm/7nm area scale: {area_scale(12, 7):.4f}')
print(f'12nm/5nm area scale: {area_scale(12, 5):.4f}')
print(f'12nm/3nm area scale: {area_scale(12, 3):.4f}')



l1_cacti = pat.cacti.cacti(32 * 1024, node=16)
print(l1_cacti)

# galileo_area(768, 512, 12)

# galileo_area(2048, 512, 12)
# galileo_area(4096 + 1024 + 256, 512, 7)

# galileo_area(3840, 512, 12)
# galileo_area(16384, 512, 5)

# galileo_area(3584, 512, 12)
# galileo_area(16384 + 8192, 512, 3)
