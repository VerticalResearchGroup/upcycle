from dataclasses import dataclass

from .common import *

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
    train_perf : int
    train_bs : int

a100_perf = {
    # https://github.com/mlcommons/inference_results_v1.0/tree/master/closed/NVIDIA/results/A100-PCIex1_TRT/bert-99/
    'bert-large-squad-avg': NvidiaAppStats(Dtype.I8, 618, 2880, 1024, Dtype.FP16, None, None),
    # 'bert-large-512': NvidiaAppStats(Dtype.I8, None, None, None, Dtype.FP16, None, None),

    # https://github.com/mlcommons/inference_results_v1.0/tree/master/closed/NVIDIA/results/A100-PCIex1_TRT/resnet50
    'resnet50': NvidiaAppStats(Dtype.I8, 2012, 31063, 2048, Dtype.FP16, None, None),

    # https://github.com/mlcommons/inference_results_v1.0/tree/master/closed/NVIDIA/results/A100-PCIex1_TRT/ssd-resnet34
    'ssdrn34-1200': NvidiaAppStats(Dtype.I8, 508, 827, 64, Dtype.FP16, None, None),

    # https://github.com/mlcommons/inference_results_v1.0/tree/master/closed/NVIDIA/results/A100-PCIex1_TRT/rnnt
    'rnnt': NvidiaAppStats(Dtype.FP16, 67, 11740, 2048, Dtype.FP16, None, None),

    # https://github.com/mlcommons/inference_results_v1.0/tree/master/closed/NVIDIA/results/A100-PCIex1_TRT/3d-unet-99
    # 'unet': NvidiaAppStats(Dtype.I8, 35, 51, 2, Dtype.FP16, None, None),
}

