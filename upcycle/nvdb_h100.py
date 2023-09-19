from dataclasses import dataclass
import yaml

from .common import *

from . import arch
from . import ops
from . import apps
from . import simdb
from . import nvdb

h100_peak = {
    Dtype.I8: 3026e12 / 2,
    Dtype.FP16: 1513e12 / 2,
}

h100_a100sw_speedup = {
    'bert-large-squad': 2.68,
    'bert-large-pretrain': 2.68,
    'bert': 2.68,
    'resnet50': 2.07,
    'ssdrn34-1200': 2.07,
    'ssdrn34-300': 2.07,
    'ssdrn34': 2.07,
    'rnnt': 1.75,
    'unet': 1.87,
    'dlrm': 1.55
}

def get_perf(appname, mode, batch):
    return nvdb.get_perf(appname, mode, batch) * h100_a100sw_speedup[appname]

def get_batch_size(appname, mode, batch):
    return nvdb.get_batch_size(appname, mode, batch)

def get_util(appname, mode, batch):
    app = nvdb.a100_perf[apps.short_appname_map[(appname, mode)]]
    dtype = app.infer_dtype if mode == 'infer' else app.train_dtype
    nvips = get_perf(appname, mode, batch)

    _, trace, _, _ = apps.workload_factory(
        apps.short_appname_map[(appname, mode)], 1, infer=mode == 'infer')

    return nvips * trace.flops / h100_peak[dtype]

def pj_per_op(appname, mode, batch):
    nvips = get_perf(appname, mode, batch)

    nv_pow_w = 350
    nv_area = 826
    _, trace, _, _ = apps.workload_factory(
        apps.short_appname_map[(appname, mode)], 1, infer=mode == 'infer')

    # nv_tops_per_mm2 = trace.flops / nv_area / 1e12
    nv_pj_per_op = nv_pow_w / nvips / trace.flops * 1e12
    return nv_pj_per_op

@dataclass(frozen=True)
class NvLayerData:
    arch : arch.Arch
    arch_ext : simdb.ArchExtConfig
    op : ops.Operator
    fp16_lat_sec : float

    @property
    def tot_cyc(self): raise NotImplementedError()

    @property
    def tot_lat(self):
        if self.op.dtype == Dtype.FP16: return self.fp16_lat_sec
        elif self.op.dtype == Dtype.I8: return self.fp16_lat_sec / 2

    @property
    def max_pow_w(self): return None

    @property
    def tot_flops(self): return self.op.flops

    @property
    def tot_energy_j(self): raise NotImplementedError()

    @property
    def util(self):
        return self.tot_flops / self.tot_lat / \
            self.arch.total_peak_compute(self.op.dtype, None)

class NvDb(simdb.SimDb):
    def __init__(self):
        self.arch = arch.H100()
        self.data = yaml.load(
            open(f'./results/{self.arch.keystr}-bench.yaml'),
            Loader=yaml.SafeLoader)

    def area_mm2(self): raise NotImplementedError()

    @functools.lru_cache(maxsize=1024)
    def __getitem__(self, x):
        op : ops.Operator
        (op, _) = x
        fp16_lat_sec = float(self.data[repr(op)])

        return NvLayerData(
            self.arch,
            None,
            op,
            fp16_lat_sec if fp16_lat_sec > 0 else 100000)

