from dataclasses import dataclass
import logging

from ...common import *
from ..common import *
from .op import *
from .mkkn import *
from .mknk import *
from .kmkn import *
from .kmnk import *

logger = logging.getLogger(__name__)

def choose_tile(arch : Arch, op : Matmul):
    tile = {
        (256, False, False, Dtype.I8):   MatmulTile256MKKNI8,
        (256, False, True,  Dtype.I8):   MatmulTile256MKNKI8_SmallK,
        (256, False, False, Dtype.FP16): MatmulTile256MKKNFP16,
        (256, False, True,  Dtype.FP16): MatmulTile256MKNKFP16_SmallK,
        (256, True,  False, Dtype.FP16): MatmulTile256KMKNFP16,
        (256, True,  True,  Dtype.FP16): MatmulTile256KMNKFP16,

        (512, False, False, Dtype.I8):   MatmulTile512MKKNI8,
        (512, False, True,  Dtype.I8):   MatmulTile512MKNKI8_SmallK,
        (512, False, False, Dtype.FP16): MatmulTile512MKKNFP16,
        (512, False, True,  Dtype.FP16): MatmulTile512MKNKFP16_SmallK,
        (512, True,  False, Dtype.FP16): MatmulTile512KMKNFP16,
        (512, True,  True,  Dtype.FP16): MatmulTile512KMNKFP16,

        (1024, False, False, Dtype.I8):   MatmulTile1024MKKNI8,
        (1024, False, True,  Dtype.I8):   MatmulTile1024MKNKI8_SmallK,
        (1024, False, False, Dtype.FP16): MatmulTile1024MKKNFP16,
        (1024, False, True,  Dtype.FP16): MatmulTile1024MKNKFP16_SmallK,
        (1024, True,  False, Dtype.FP16): MatmulTile1024KMKNFP16,
        (1024, True,  True,  Dtype.FP16): MatmulTile1024KMNKFP16,
    }[(arch.vbits, op.tr_a, op.tr_b, op.dtype)]

    logger.debug(f'Tile for arch={arch} and op={op}: {tile}')
    return tile

@M.register_placement(
    [OracleArch, BgroupArch, FbcastArch, HierArch, CoarseOracle],
    [Matmul, Linear])
def place_matmul_default(arch : Arch, mm : Matmul, sim : M.SimBase):
    ins, outs = mm.make_tensors(arch)
    tile = choose_tile(arch, mm)

    sim.map2d_place([
        [
            [
                tile(arch, mm, ins, outs, False, li, bm1, bn1, bk1)
                for li in Slice(0, mm.l).indices
                for bk1 in Slice(0, mm.k).subslice(tile.tk * 4)
                for bm1 in bm0.subslice(tile.tm * 4)
                for bn1 in bn0.subslice(tile.tn * 4)
            ]
            for bm0 in Slice(0, mm.m).blkslice(64)
        ]
        for bn0 in Slice(0, mm.n).blkslice(32)
    ])
