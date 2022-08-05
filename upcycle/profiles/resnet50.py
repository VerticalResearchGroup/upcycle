from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)



# @ops.placement_profile(
#     [OracleArch, BgroupArch, FbcastArch],
#     ops.Conv2D(None, None, None, 224, 224, 3, 112, 112, 64, 7, 7, 2, None, None))
# def place_rn50_l1(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
#     return ops.conv2d.place_conv_pq_spatial(arch, conv, sim)



@ops.placement_profile(
    [BgroupArch, FbcastArch], # Apparently this doesn't actually improve perf. for oracle.
    ops.Conv2D(None, None, None, None, None, None, None, None, None, 1, 1, None, None, None))
def place_conv_1x1(arch : Arch, conv : ops.Conv2D, sim : M.SimBase):
    # We observe in this case the convolution degenerates into a large matmul.
    mm = ops.Matmul(
        conv.dtype, conv.train, 1, conv.n * conv.p * conv.q, conv.k, conv.c,
        False, not conv.tr_w)

    assert mm.flops == conv.flops
    return M.place_op('pg', arch, mm, sim, False)

