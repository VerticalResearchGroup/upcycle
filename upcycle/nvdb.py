from dataclasses import dataclass
import yaml

from .common import *

from . import arch
from . import ops
from . import apps
from . import simdb

# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
a100_peak = {
    Dtype.I8: 624e12,
    Dtype.FP16: 312e12,
}

@dataclass(frozen=True)
class NvidiaAppStats:
    infer_dtype : Dtype
    infer_online_perf : int
    infer_offline_perf : int
    infer_offline_bs : int

    train_dtype : Dtype
    train_large_perf : int
    train_large_bs : int
    train_small_perf : int
    train_small_bs : int


a100_perf = {
    # NOTE: Inference Only
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/bert-99/Offline/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/bert-99/SingleStream/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/configs/bert/Offline/__init__.py
    'bert-large-squad': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=681.21,
        infer_offline_perf=3206.7, infer_offline_bs=1024,
        train_dtype=Dtype.FP16,
        train_large_perf=None, train_large_bs=None,
        train_small_perf=None, train_small_bs=None),

    # NOTE: Training Only
    'bert-large-pretrain': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=None,
        infer_offline_perf=None, infer_offline_bs=None,
        train_dtype=Dtype.FP16,
        train_large_perf=2193/8, train_large_bs=56,
        train_small_perf=2, train_small_bs=1),

    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/resnet50/Offline/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/resnet50/SingleStream/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/configs/resnet50/Offline/__init__.py
    # https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/results/dgxa100_ngc21.09_mxnet/resnet/result_0.txt
    'resnet50': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=2049.21,
        infer_offline_perf=36864.6, infer_offline_bs=2048,
        train_dtype=Dtype.FP16,
        train_large_perf=26325/8, train_large_bs=408,
        train_small_perf=193, train_small_bs=1),

    # NOTE: Inference Only
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/ssd-resnet34/Offline/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/ssd-resnet34/SingleStream/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/configs/ssd-resnet34/Offline/__init__.py
    'ssdrn34-1200': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=524.77,
        infer_offline_perf=913.68, infer_offline_bs=64,
        train_dtype=Dtype.FP16,
        train_large_perf=None, train_large_bs=None,
        train_small_perf=None, train_small_bs=None),

    # NOTE: Training Only
    # https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/results/dgxa100_ngc21.09_mxnet/ssd/result_0.txt
    'ssdrn34-300': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=None,
        infer_offline_perf=None, infer_offline_bs=None,
        train_dtype=Dtype.FP16,
        train_large_perf=12214/8, train_large_bs=112,
        train_small_perf=83, train_small_bs=1),

    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/rnnt/Offline/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/rnnt/SingleStream/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/configs/rnnt/Offline/__init__.py
    # https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/results/dgxa100_ngc21.09_pytorch/rnnt/result_0.txt
    'rnnt': NvidiaAppStats(
        infer_dtype=Dtype.FP16,
        infer_online_perf=72.06,
        infer_offline_perf=13594.4, infer_offline_bs=2048,
        train_dtype=Dtype.FP16,
        train_large_perf=7709/8, train_large_bs=2048/8,
        train_small_perf=38, train_small_bs=2),

    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/3d-unet-99/Offline/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/3d-unet-99/SingleStream/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/configs/3d-unet/Offline/__init__.py
    'unet': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=2.98 * apps.kits19_patches_per_sample,
        infer_offline_perf=2.99 * apps.kits19_patches_per_sample, infer_offline_bs=1,
        train_dtype=Dtype.FP16,
        train_large_perf=283/8, train_large_bs=56/8,
        train_small_perf=31, train_small_bs=1),

    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/configs/dlrm/Offline/__init__.py
    # https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/results/dgxa100_ngc21.09_merlin_hugectr/dlrm/result_0.txt
    'dlrm': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=600183.32,
        infer_offline_perf=2.47727e+06, infer_offline_bs=315000,
        train_dtype=Dtype.FP16,
        train_large_perf=36895473/8, train_large_bs=55296,
        train_small_perf=None, train_small_bs=None),
}

def get_perf(appname, mode, batch):
    app = a100_perf[apps.short_appname_map[(appname, mode)]]

    if mode == 'infer':
        if batch == 'online':
            result = app.infer_online_perf
        elif batch == 'offline':
            result = app.infer_offline_perf
        else: assert False

    elif mode == 'train':
        if batch == 'large':
            result = app.train_large_perf
        elif batch == 'small':
            result = app.train_small_perf
        else: assert False

    assert result is not None, f'No perf for {appname} {mode} {batch}'
    return result

def get_batch_size(appname, mode, batch):
    app = a100_perf[apps.short_appname_map[(appname, mode)]]

    if mode == 'infer':
        if batch == 'online':
            result = 1
        elif batch == 'offline':
            result = app.infer_offline_bs
        else: assert False

    elif mode == 'train':
        if batch == 'large':
            result = app.train_large_bs
        elif batch == 'small':
            result = app.train_small_bs
        else: assert False

    assert result is not None, f'No batch size for {appname} {mode} {batch}'
    return result

def get_util(appname, mode, batch):
    app = a100_perf[apps.short_appname_map[(appname, mode)]]
    dtype = app.infer_dtype if mode == 'infer' else app.train_dtype
    nvips = get_perf(appname, mode, batch)

    _, trace, _, _ = apps.workload_factory(
        apps.short_appname_map[(appname, mode)], 1, infer=mode == 'infer')

    nv_peak_compute = {
        Dtype.I8: 624e12,
        Dtype.FP16: 624e12/2,
    }

    if appname == 'rnnt' and mode == 'infer' and batch == 'offline':
        print(nvips, dtype, trace.flops, nv_peak_compute[dtype])

    return nvips * trace.flops / nv_peak_compute[dtype]

def pj_per_op(appname, mode, batch):
    nvips = get_perf(appname, mode, batch)

    nv_pow_w = 300
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
    def max_pow_w(self): return 300

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
        self.arch = arch.A100()
        self.data = yaml.load(
            open(f'./results/{self.arch.keystr}-bench.yaml'),
            Loader=yaml.SafeLoader)

    def area_mm2(self): return 826

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

