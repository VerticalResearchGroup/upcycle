from dataclasses import dataclass
import numpy as np
import logging

from ..common import *
from ..arch import *
from .. import model as M
from .. import ops

logger = logging.getLogger(__name__)

@M.register_placement( # Layer 17
    [OracleArch, BgroupArch, FbcastArch, HierArch],
    ops.Matmul(None, None, None, 27, 27, 128, None, None))
def place_dlrm_interaction(arch : Arch, mm : ops.Matmul, sim : M.SimBase):
    l = mm.l
    m = mm.m
    n = mm.n
    k = mm.k

    a = M.Tensor(arch, 1, mm.dtype, (l, m, k) if not mm.tr_a else (l, k, m))
    b = M.Tensor(arch, 2, mm.dtype, (l, k, n) if not mm.tr_b else (l, n, k))
    c = M.Tensor(arch, 3, mm.dtype, (l, m, n))

    tile = {
        (False, False, Dtype.I8): ops.matmul.MatmulTileMKKNI8,
        (False, True, Dtype.I8): ops.matmul.MatmulTileMKNKI8,
        (False, False, Dtype.FP16): ops.matmul.MatmulTileMKKNFP16,
        (False, True, Dtype.FP16): ops.matmul.MatmulTileMKNKFP16,
        (True, False, Dtype.FP16): ops.matmul.MatmulTileKMKNFP16,
        (True, True, Dtype.FP16): ops.matmul.MatmulTileKMNKFP16,
    }[(mm.tr_a, mm.tr_b, mm.dtype)]

    sim.map2d_place([
        [
            [
                tile(arch, mm, [a, b], [c], False, li, bm1, bn1, bk1)
                for li in bl1.indices
                for bm1 in Slice(0, mm.m).subslice(tile.tm)
                for bn1 in Slice(0, mm.n).subslice(tile.tn)
                for bk1 in Slice(0, mm.k).blkslice(1)
            ]
            for bl1 in bl0.blkslice(64)
        ]
        for bl0 in Slice(0, mm.l).blkslice(32)
    ])
