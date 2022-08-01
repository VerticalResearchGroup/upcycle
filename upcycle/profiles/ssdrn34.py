from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)


@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 300, 300, 64, 150, 150, 128, 3, 3, 2, None, None))
def place_ssdrn34_l7(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 150, 150, 128, 150, 150, 128, 3, 3, 1, None, None))
def place_ssdrn34_l8(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)


@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 150, 150, 128, 150, 150, 256, 3, 3, 1, None, None))
def place_ssdrn34_l16(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 150, 150, 256, 150, 150, 256, 3, 3, 1, None, None))
def place_ssdrn34_l17(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 150, 150, 256, 75, 75, 512, 3, 3, 2, None, None))
def place_ssdrn34_l30(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 37/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=9, w=9, c=256, p=9, q=9, k=128, r=1, s=1, stride=1, pad=0, tr_w=False)
# TODO: place_conv_pq_spatial is worse, find better strategy
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 19, 19, 128, 9, 9, 256, 3, 3, 2, None, None))
# def place_ssdrn34_l36(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 38/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=9, w=9, c=128, p=7, q=7, k=256, r=3, s=3, stride=1, pad=0, tr_w=False)
# TODO: place_conv_pq_spatial is worse, find better strategy
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 9, 9, 256, 9, 9, 128, 1, 1, 1, None, None))
# def place_ssdrn34_l37(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# TODO: place_conv_pq_spatial is worse, find better strategy
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 9, 9, 128, 7, 7, 256, 3, 3, 1, None, None))
# def place_ssdrn34_l38(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 39/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=150, w=150, c=256, p=50, q=50, k=16, r=3, s=3, stride=3, pad=1, tr_w=False)
# TODO: place_conv_pq_spatial is worse, find better strategy
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 150, 150, 256, 50, 50, 16, 3, 3, 3, None, None))
# def place_ssdrn34_l39(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 40/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=150, w=150, c=256, p=50, q=50, k=324, r=3, s=3, stride=3, pad=1, tr_w=False)
@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 150, 150, 256, 50, 50, 324, 3, 3, 3, None, None))
def place_ssdrn34_l40(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 41/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=75, w=75, c=512, p=25, q=25, k=24, r=3, s=3, stride=3, pad=1, tr_w=False)
# BUG: place_conv_pq_spatial barfs for this shape
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 75, 75, 512, 25, 25, 24, 3, 3, 3, None, None))
# def place_ssdrn34_l41(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 42/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=75, w=75, c=512, p=25, q=25, k=486, r=3, s=3, stride=3, pad=1, tr_w=False)
# TODO: place_conv_pq_spatial is worse, find better strategy
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 75, 75, 512, 25, 25, 486, 3, 3, 3, None, None))
# def place_ssdrn34_l42(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 43/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=38, w=38, c=512, p=13, q=13, k=24, r=3, s=3, stride=3, pad=1, tr_w=False)
# BUG: place_conv_pq_spatial barfs for this shape
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 38, 38, 512, 13, 13, 24, 3, 3, 3, None, None))
# def place_ssdrn34_l43(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 44/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=38, w=38, c=512, p=13, q=13, k=486, r=3, s=3, stride=3, pad=1, tr_w=False)
# TODO: place_conv_pq_spatial is worse, find better strategy
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 38, 38, 512, 13, 13, 486, 3, 3, 3, None, None))
# def place_ssdrn34_l44(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 45/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=19, w=19, c=256, p=7, q=7, k=24, r=3, s=3, stride=3, pad=1, tr_w=False)
# BUG: place_conv_pq_spatial barfs for this shape
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 19, 19, 256, 7, 7, 24, 3, 3, 3, None, None))
# def place_ssdrn34_l45(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 46/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=19, w=19, c=256, p=7, q=7, k=486, r=3, s=3, stride=3, pad=1, tr_w=False)
@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 19, 19, 256, 7, 7, 486, 3, 3, 3, None, None))
def place_ssdrn34_l46(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 47/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=9, w=9, c=256, p=3, q=3, k=16, r=3, s=3, stride=3, pad=1, tr_w=False)
# TODO: place_conv_pq_spatial is worse, find better strategy
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 9, 9, 256, 3, 3, 16, 3, 3, 3, None, None))
# def place_ssdrn34_l47(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 48/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=9, w=9, c=256, p=3, q=3, k=324, r=3, s=3, stride=3, pad=1, tr_w=False)
@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 9, 9, 256, 3, 3, 324, 3, 3, 3, None, None))
def place_ssdrn34_l48(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 49/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=7, w=7, c=256, p=3, q=3, k=16, r=3, s=3, stride=3, pad=1, tr_w=False)
# TODO: place_conv_pq_spatial is worse, find better strategy
# @ops.placement_profile(
#     [OracleArch, BgroupArch],
#     ops.Conv2D(None, None, None, 7, 7, 256, 3, 3, 16, 3, 3, 3, None, None))
# def place_ssdrn34_l49(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)

# Layer 50/51: Conv2D(dtype=<Dtype.I8: 1>, train=True, n=1, h=7, w=7, c=256, p=3, q=3, k=324, r=3, s=3, stride=3, pad=1, tr_w=False)
@ops.placement_profile(
    [OracleArch, BgroupArch, FbcastArch],
    ops.Conv2D(None, None, None, 7, 7, 256, 3, 3, 324, 3, 3, 3, None, None))
def place_ssdrn34_l50(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)
