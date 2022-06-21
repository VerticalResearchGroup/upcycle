from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)



@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 224, 224, 3, 112, 112, 64, 7, 7, 2, None, None))
def place_rn50_l1(arch : Arch, conv : ops.Conv2D):
    return ops.conv2d.place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 56, 56, 128, 28, 28, 128, 3, 3, 2, None, None))
def place_rn50_l12(arch : Arch, conv : ops.Conv2D):
    return ops.conv2d.place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, None, 28, 28, 128, 28, 28, 128, 3, 3, 1, None, None))
def place_rn50_l16(arch : Arch, conv : ops.Conv2D):
    return ops.conv2d.place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, Slice(2, 256), 28, 28, 512, 28, 28, 128, 1, 1, 1, None, None))
def place_rn50_l15b(arch : Arch, conv : ops.Conv2D):
    return ops.conv2d.place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, Slice(2, 256), 14, 14, 1024, 14, 14, 256, 1, 1, 1, None, None))
def place_rn50_l37b(arch : Arch, conv : ops.Conv2D):
    return ops.conv2d.place_conv_pq_spatial(arch, conv)

@ops.placement_profile(
    [OracleArch, BgroupArch],
    ops.Conv2D(None, None, Slice(2, 256), 14, 14, 256, 14, 14, 256, 3, 3, 1, None, None))
def place_rn50_l38b(arch : Arch, conv : ops.Conv2D):
    return ops.conv2d.place_conv_pq_spatial(arch, conv)

# Layer 46/106: Conv2D(dtype=<Dtype.I8: 2>, train=True, n=1, h=14, w=14, c=1024, p=7, q=7, k=2048, r=1, s=1, stride=2, pad=0, tr_w=True)
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 14, 14, 1024, 7, 7, 2048, 1, 1, 2, None, None))
# def place_rn50_l46(arch : Arch, conv : ops.Conv2D):
#     return ops.conv2d.place_conv_pq_spatial2(arch, conv)
