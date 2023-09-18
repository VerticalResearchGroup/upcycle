
from .. import pat


def area_scale(from_node, to_node):
    factors = {
        16: 1.0,
        12: 0.86,
        7: 0.34,
        5: 0.19,
        3: 0.19/1.5,
    }

    return factors[from_node] / factors[to_node]


num_tiles = 2048

int8_mac_7nm = pat.reflib.int8_mac_7nm.area
vrf_1k_7nm = pat.reflib.vrf_1k_7nm.area
core_7nm = pat.reflib.core_7nm.area
pe_core_7nm = (int8_mac_7nm * 3 * (512 // 8)) + \
    vrf_1k_7nm * 2 + core_7nm

l1_cacti = pat.cacti.cacti(32 * 1024)
l2_cacti = pat.cacti.cacti(16 * 1024)
llc_cacti = pat.cacti.cacti(32 * 2**20 / num_tiles)


l1_7nm = l1_cacti.area_mm2
l2_slice_area = l2_cacti.area_mm2
llc_slice_area = llc_cacti.area_mm2
pe_area = pe_core_7nm + l1_7nm + l2_slice_area + llc_slice_area

print(f'PE area: {pe_area:.4f} mm^2')
print(f'    Core: {pe_core_7nm:.4f} mm^2')
print(f'    L1: {l1_7nm:.4f} mm^2')
print(f'    L2 slice: {l2_slice_area:.4f} mm^2')
print(f'    LLC slice: {llc_slice_area:.4f} mm^2')

print(f'Total area: {pe_area * num_tiles:.4f} mm^2 (7nm)')
print(f'Total area: {pe_area * num_tiles / area_scale(7, 12):.4f} mm^2 (12nm)')


