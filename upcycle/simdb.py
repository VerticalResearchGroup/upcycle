import yaml
import functools
from ast import literal_eval as make_tuple

from .common import *
from .arch import *
from . import ops
from . import apps
from . import pat

pow_c = 1.2e-9

power_table = {
    1.0e9: 1,
    1.1e9: 1.1,
    1.2e9: 1.391715976,
    1.3e9: 1.507692308,
    1.4e9: 1.623668639,
    1.5e9: 1.73964497,
    1.6e9: 1.855621302,
    1.7e9: 1.971597633,
    1.8e9: 2.239349112,
    1.9e9: 2.363757396,
    2.0e9: 2.843195266,
    2.1e9: 2.98535503,
    2.2e9: 3.332544379,
    2.3e9: 3.705177515,
    2.4e9: 4.104142012,
    2.5e9: 4.530325444,
    2.6e9: 4.984615385,
    2.7e9: 5.767455621,
    2.8e9: 6.3,
    2.9e9: 6.863905325,
    3.0e9: 8.205621302
}

@dataclass(frozen=True)
class ArchExtConfig:
    freq : float
    membw : float
    compute_scale : float = 1.0
    noc_scale : float = 1.0
    mem_scale : float = 1.0

@dataclass(frozen=True)
class LayerData:
    arch_ext : ArchExtConfig
    op : ops.Operator
    onchip_cycles : int
    mem_cyc : int
    l1_accesses : int
    l2_accesses : int
    llc_accesses : int
    total_read_bytes : int
    total_weight_bytes : int
    total_write_bytes : int
    l1_energy_j : float
    l2_energy_j : float
    llc_energy_j : float
    power_w : float
    energy_j : float
    util : float

    @property
    def real_cyc(self):
        return max(self.onchip_cycles, self.mem_cyc, 1)

    @property
    def real_lat(self):
        return self.real_cyc / self.arch_ext.freq

    @property
    def mem_bound(self): return self.onchip_cycles < self.mem_cyc

class SimDb:
    def __init__(self, arch : HierArch):
        self.arch = arch
        self.data = yaml.load(
            open(f'./results/{arch.keystr}.yaml'),
            Loader=yaml.SafeLoader)

        self.l1_cacti = pat.cacti.cacti(arch.l1.capacity)
        self.l2_cacti = pat.cacti.cacti(arch.l2.capacity // arch.tiles_per_group)
        self.llc_cacti = pat.cacti.cacti(128 * 2**20 // arch.ntiles)

    def area_mm2(self):
        int8_mac_7nm = pat.reflib.int8_mac_7nm.area
        vrf_1k_7nm = pat.reflib.vrf_1k_7nm.area
        core_7nm = pat.reflib.core_7nm.area
        pe_core_7nm = (int8_mac_7nm * 3 * (self.arch.vbits // 8)) + \
            vrf_1k_7nm * 2 + core_7nm

        l1_7nm = self.l1_cacti.area_mm2
        l2_slice_area = self.l2_cacti.area_mm2
        llc_slice_area = self.llc_cacti.area_mm2
        pe_area = pe_core_7nm + l1_7nm + l2_slice_area + llc_slice_area
        return pe_area * self.arch.ntiles

    def layer_power_w(self, yd, cyc : int, cfg : ArchExtConfig):
        leakage_w = (
            self.l1_cacti.leak_w +
            self.l2_cacti.leak_w +
            self.llc_cacti.leak_w
        ) * self.arch.ntiles

        pow_scale = power_table[cfg.freq] / power_table[2e9]
        core_w = pat.reflib.core_7nm.power

        l1_j = yd['l1_accesses'] * self.l1_cacti.read_j
        l2_j = yd['l2_accesses'] * self.l2_cacti.read_j
        llc_j = yd['llc_accesses'] * self.llc_cacti.read_j
        cache_w = (l1_j + l2_j + llc_j) / (cyc / cfg.freq)

        return \
            leakage_w + \
            core_w * self.arch.ntiles + \
            cache_w * pow_scale + \
            pat.reflib.netcon_w * self.arch.ntiles + \
            pat.reflib.hbm_w

    @functools.lru_cache(maxsize=1024)
    def __getitem__(self, x):
        op : ops.Operator
        cfg : ArchExtConfig
        (op, cfg) = x
        yd = self.data[repr(op)]
        mem_bytes_per_cycle = cfg.mem_scale * cfg.membw / cfg.freq
        offchip_bytes = max(yd['total_read_bytes'], yd['total_weight_bytes'])
        mem_cyc = int(0 if mem_bytes_per_cycle == 0 else offchip_bytes / mem_bytes_per_cycle)
        si = self.arch.lookup_scale(cfg.compute_scale, cfg.noc_scale)
        layer_cyc = make_tuple(yd['cycles'])

        real_cyc = max(layer_cyc[si], mem_cyc, 1)
        power_w = self.layer_power_w(yd, real_cyc, cfg)
        energy_j = power_w * real_cyc / cfg.freq
        util = op.flops / real_cyc / self.arch.ntiles / self.arch.peak_opc(op.dtype)

        return LayerData(
            cfg,
            op,
            int(layer_cyc[si]),
            mem_cyc,
            yd['l1_accesses'],
            yd['l2_accesses'],
            yd['llc_accesses'],
            yd['total_read_bytes'],
            yd['total_weight_bytes'],
            yd['total_write_bytes'],
            yd['l1_accesses'] * self.l1_cacti.read_j,
            yd['l2_accesses'] * self.l2_cacti.read_j,
            yd['llc_accesses'] * self.llc_cacti.read_j,
            power_w,
            energy_j,
            util)

    def trace(self, app : apps.Trace, cfg : ArchExtConfig) -> list[LayerData]:
        return [self[op, cfg] for op in app.oplist]

    def lat(self, app : apps.Trace, cfg : ArchExtConfig) -> float:
        cyc = sum([ld.real_cyc for ld in self.trace(app, cfg)])
        return cyc / cfg.freq

    def perf(self, app : apps.Trace, cfg : ArchExtConfig) -> float:
        cyc = sum([ld.real_cyc for ld in self.trace(app, cfg)])
        return app.bs * cfg.freq / cyc

    def pj_per_op(self, app : apps.Trace, cfg : ArchExtConfig) -> list[float]:
        return sum(ld.energy_j for ld in self.trace(app, cfg)) / app.flops * 1e12

    def tops_per_mm2(self, app : apps.Trace, cfg : ArchExtConfig) -> float:
        # print(f'{app.bs} {app.flops} {self.lat(app, cfg)} {self.area_mm2()}')
        return app.flops / self.lat(app, cfg) / self.area_mm2() / 1e12

@functools.lru_cache(maxsize=1024)
def cached_simdb(arch : HierArch):
    return SimDb(arch)
