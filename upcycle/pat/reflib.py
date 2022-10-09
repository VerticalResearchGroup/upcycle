from .common import *

int8_mac_28nm = Component(32, 336.672e-6, 1 / 1e9, 283e-6)
int8_mac_7nm = Component(32, 336.672e-6, 1 / 1e9, 283e-6).scale_isofreq(7)

vrf_1k_22nm_proto = Component(20, 20000e-12, 1 / 1e9, 1.0)
vrf_1k_7nm_proto = Component(20, 20000e-12, 1 / 1e9, 1.0).scale_isofreq(7)

vrf_1k_7nm = Component(
    7,
    vrf_1k_7nm_proto.area,
    vrf_1k_7nm_proto.delay,
    vrf_1k_7nm_proto.power * 1.3)

core_45nm = Component(45, 0.044, 1 / 1e9, 14e-3)
core_7nm = Component(45, 0.044, 1 / 1e9, 14e-3).scale_isofreq(7)

netcon_16nm = Component(16, 1.0, 1 / 1e9, 65e-3)
netcon_7nm = Component(16, 1.0, 1 / 1e9, 65e-3).scale_isofreq(7)

# We are assuming HBM2 takes ~24 W
hbm_w = 24
netcon_w = 30/4/1000
