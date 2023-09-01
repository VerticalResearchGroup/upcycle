from numpy import real_if_close
import yaml
import functools
from ast import literal_eval as make_tuple

from .common import *
from .arch import *
from . import ops
from . import apps
from . import pat

logger = logging.getLogger(__name__)

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
    arch : Arch
    arch_ext : ArchExtConfig
    ops : tuple[ops.Operator]
    onchip_cycles : tuple[int]
    mem_cycs : tuple[int]
    powers_w : tuple[float]
    energies_j : tuple[float]

    @property
    def tot_cyc(self):
        return sum(
            max(self.onchip_cycles[i], self.mem_cycs[i], 1)
            for i in range(len(self.onchip_cycles)))

    @property
    def tot_lat(self): return self.tot_cyc / self.arch_ext.freq

    @property
    def max_pow_w(self): return max(self.powers_w)

    @property
    def tot_flops(self): return sum(op.flops for op in self.ops)

    @property
    def tot_energy_j(self): return sum(self.energies_j)

    @property
    def util(self):
        if max(self.onchip_cycles) == 0:
            return 0

        return self.tot_flops / self.tot_lat / \
            self.arch.total_peak_compute(self.ops[0].dtype, self.arch_ext.freq)

    def __add__(self, other : 'LayerData') -> 'LayerData':
        return LayerData(
            self.arch,
            self.arch_ext,
            self.ops + other.ops,
            self.onchip_cycles + other.onchip_cycles,
            self.mem_cycs + other.mem_cycs,
            self.powers_w + other.powers_w,
            self.energies_j + other.energies_j)

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

    def _layer_power_w(self, yd, cyc : int, cfg : ArchExtConfig):
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
            pat.reflib.hbm_w / 2

    @functools.lru_cache(maxsize=1024)
    def __getitem__(self, x):
        op : ops.Operator
        cfg : ArchExtConfig
        (op, cfg) = x

        if isinstance(op, apps.TrainOp):
            result = None
            for opp in op:
                if result is None:
                    result = self[(opp, cfg)]
                else:
                    result += self[(opp, cfg)]
            return result

        if repr(op) == 'Conv2D[Int8](n=1, i=(300, 300)x3+3 w=(7, 7)x64x3 o=(150, 150)x64 by 2)' and \
            'Conv2D[Int8](n=1, i=(300, 300)x3+3 w=(7, 7)x64x3 o=(150, 150)x64 by 2)' not in self.data:
            yd = self.data['Conv2D[Int8](n=1, i=(1200, 1200)x3+3 w=(7, 7)x64x3 o=(600, 600)x64 by 2)']

            yd['cycles'] = str(tuple(x * 4 for x in make_tuple(yd['cycles'])))
            yd['total_read_bytes'] = float(yd['total_read_bytes']) * 4
            yd['total_weight_bytes'] = float(yd['total_weight_bytes']) * 4
            yd['total_write_bytes'] = float(yd['total_write_bytes']) * 4
            yd['l1_accesses'] = float(yd['l1_accesses']) * 4
            yd['l2_accesses'] = float(yd['l2_accesses']) * 4
            yd['llc_accesses'] = float(yd['llc_accesses']) * 4


        elif repr(op) == 'Conv2D[Int8](n=16, i=(300, 300)x3+3 w=(7, 7)x64x3 o=(150, 150)x64 by 2)' and \
            'Conv2D[Int8](n=16, i=(300, 300)x3+3 w=(7, 7)x64x3 o=(150, 150)x64 by 2)' not in self.data:
            yd = self.data['Conv2D[Int8](n=16, i=(1200, 1200)x3+3 w=(7, 7)x64x3 o=(600, 600)x64 by 2)']

            yd['cycles'] = str(tuple(x * 4 for x in make_tuple(yd['cycles'])))
            yd['total_read_bytes'] = float(yd['total_read_bytes']) * 4
            yd['total_weight_bytes'] = float(yd['total_weight_bytes']) * 4
            yd['total_write_bytes'] = float(yd['total_write_bytes']) * 4
            yd['l1_accesses'] = float(yd['l1_accesses']) * 4
            yd['l2_accesses'] = float(yd['l2_accesses']) * 4
            yd['llc_accesses'] = float(yd['llc_accesses']) * 4

        else:
            try:
                yd = self.data[repr(op)]
            except KeyError:
                logger.error(f'No data for {repr(op)}')
                logger.error(f'Arch: {self.arch}')
                return LayerData(
                    self.arch,
                    cfg,
                    (op, ),
                    (0, ),
                    (0, ),
                    (0, ),
                    (0, ))

        mem_bytes_per_cycle = cfg.mem_scale * cfg.membw / cfg.freq
        offchip_bytes = max(yd['total_read_bytes'], yd['total_weight_bytes'])
        mem_cyc = int(0 if mem_bytes_per_cycle == 0 else offchip_bytes / mem_bytes_per_cycle)
        si = self.arch.lookup_scale(cfg.compute_scale, cfg.noc_scale)
        layer_cyc = make_tuple(yd['cycles'])

        real_cyc = max(layer_cyc[si], mem_cyc, 1)
        power_w = self._layer_power_w(yd, real_cyc, cfg)
        energy_j = power_w * real_cyc / cfg.freq

        return LayerData(
            self.arch,
            cfg,
            (op, ),
            (int(layer_cyc[si]), ),
            (mem_cyc, ),
            (power_w, ),
            (energy_j, ))

    def trace(self, app : apps.Trace, cfg : ArchExtConfig) -> list[LayerData]:
        return [self[op, cfg] for op in app.oplist]

    def lat(self, app : apps.Trace, cfg : ArchExtConfig) -> float:
        cyc = sum([ld.tot_cyc for ld in self.trace(app, cfg)])
        return cyc / cfg.freq

    def perf(self, app : apps.Trace, cfg : ArchExtConfig) -> float:
        cyc = sum([ld.tot_cyc for ld in self.trace(app, cfg)])
        return app.bs * cfg.freq / cyc

    def util(self, app : apps.Trace, cfg : ArchExtConfig) -> float:
        return self.perf(app, cfg) * app.flops / app.bs / self.arch.total_peak_compute(app.oplist[0].dtype, cfg.freq)

    def pj_per_op(self, app : apps.Trace, cfg : ArchExtConfig) -> list[float]:
        return sum(ld.tot_energy_j for ld in self.trace(app, cfg)) / app.flops * 1e12

    def power(self, app : apps.Trace, cfg : ArchExtConfig) -> list[float]:
        return sum(ld.tot_energy_j for ld in self.trace(app, cfg)) / self.lat(app, cfg)

    def tops_per_mm2(self, app : apps.Trace, cfg : ArchExtConfig) -> float:
        # print(f'{app.bs} {app.flops} {self.lat(app, cfg)} {self.area_mm2()}')
        return app.flops / self.lat(app, cfg) / self.area_mm2() / 1e12

@functools.lru_cache(maxsize=1024)
def cached_simdb(arch : HierArch):
    return SimDb(arch)
