from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)



@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 224, 224, 3, 112, 112, 64, 7, 7, 2, None, None))
def place_rn50_l1(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 56, 56, 128, 28, 28, 128, 3, 3, 2, None, None))
def place_rn50_l12(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 15/53: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=28, w=28, c=512, p=28, q=28, k=128, r=1, s=1, stride=1, pad=0, tr_w=False)
@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 28, 28, 512, 28, 28, 128, 1, 1, 1, None, None))
def place_rn50_l15(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 28, 28, 128, 28, 28, 128, 3, 3, 1, None, None))
def place_rn50_l16(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 28/53: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=14, w=14, c=1024, p=14, q=14, k=256, r=1, s=1, stride=1, pad=0, tr_w=False)
@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 14, 14, 1024, 14, 14, 256, 1, 1, 1, None, None))
def place_rn50_l28(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 44/53: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=14, w=14, c=512, p=7, q=7, k=512, r=3, s=3, stride=2, pad=1, tr_w=False)
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, Slice(2, 128), 14, 14, 512, 7, 7, 512, 3, 3, 2, None, None))
# def place_rn50_l44(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 45/53: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=7, w=7, c=512, p=7, q=7, k=2048, r=1, s=1, stride=1, pad=0, tr_w=False)
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 7, 7, 512, 7, 7, 2048, 1, 1, 1, None, None))
# def place_rn50_l45(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 46/53: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=14, w=14, c=1024, p=7, q=7, k=2048, r=1, s=1, stride=2, pad=0, tr_w=False)

# Layer 47/53: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=7, w=7, c=2048, p=7, q=7, k=512, r=1, s=1, stride=1, pad=0, tr_w=False)
