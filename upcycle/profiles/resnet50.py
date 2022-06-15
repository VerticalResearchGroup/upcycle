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
