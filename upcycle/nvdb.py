from dataclasses import dataclass

from .common import *

from . import apps

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
        train_small_perf=None, train_small_bs=None),

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
        train_small_perf=None, train_small_bs=None),

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
        train_small_perf=None, train_small_bs=None),

    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/rnnt/Offline/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/rnnt/SingleStream/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/configs/rnnt/Offline/__init__.py
    # https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/results/dgxa100_ngc21.09_pytorch/rnnt/result_0.txt
    'rnnt': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=72.06,
        infer_offline_perf=13594.4, infer_offline_bs=2048,
        train_dtype=Dtype.FP16,
        train_large_perf=7709/8, train_large_bs=2048/8,
        train_small_perf=None, train_small_bs=None),

    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/3d-unet-99/Offline/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/results/A100-PCIe-80GBx1_TRT/3d-unet-99/SingleStream/performance/run_1/mlperf_log_summary.txt
    # https://github.com/mlcommons/inference_results_v2.0/blob/master/closed/NVIDIA/configs/3d-unet/Offline/__init__.py
    'unet': NvidiaAppStats(
        infer_dtype=Dtype.I8,
        infer_online_perf=2.98 * apps.kits19_patches_per_sample,
        infer_offline_perf=2.99 * apps.kits19_patches_per_sample, infer_offline_bs=1,
        train_dtype=Dtype.FP16,
        train_large_perf=283/8, train_large_bs=56/8,
        train_small_perf=None, train_small_bs=None),

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

