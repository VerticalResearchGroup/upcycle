from dataclasses import dataclass, field
from enum import IntEnum
import functools
import numpy as np
import random
import logging

from .common import *
from .import gilbert

logger = logging.getLogger(__name__)

class TileMapping(IntEnum):
    AFFINE = 0
    HILBERT = 1

    def __repr__(self):
        return {
            TileMapping.AFFINE: 'affine',
            TileMapping.HILBERT: 'hilbert'
        }[self]

    @staticmethod
    def from_str(s):
        return {
            'affine': TileMapping.AFFINE,
            'hilbert': TileMapping.HILBERT
        }[s]

@dataclass(frozen=True)
class CacheParams:
    nbanks : int = 1
    capacity : int = 16384
    assoc : int = 16
    rports : int = 2

    def __repr__(self):
        return f'Cache({self.capacity/2**10} KB, {self.nbanks}-bank {self.assoc}-way)'

    def nset(self, line_size): return int(self.capacity / line_size / self.assoc)
    def lbits(self, line_size): return int(np.ceil(np.log2(line_size)))

    @staticmethod
    def from_str(s): return CacheParams(*[int(x) for x in s.split(',')])

@dataclass(order=True, frozen=True)
class Arch:
    """Base class for all UPCYCLE architectures.
    """

    freq : float = None
    vbits : int = None
    macs : int = None
    nrows : int = None
    ncols : int = None
    mapping : TileMapping = None
    perfect_compute : bool = None
    noc_ports_per_dir : int = None
    line_size : int = None
    l1 : CacheParams = None

    @functools.cached_property
    def vbytes(self): return self.vbits // 8

    def __post_init__(self):
        if self.noc_ports_per_dir > 1:
            logger.warn(f'Arch has {self.noc_ports_per_dir} ports per direction (>1)')

        if self.mapping == TileMapping.AFFINE:
            idmap = [
                (r, c)
                for r in range(self.nrows)
                for c in range(self.ncols)
            ]
        elif self.mapping == TileMapping.HILBERT:
            idmap = list(gilbert.gilbert2d(self.nrows, self.ncols))

        object.__setattr__(self, 'idmap', idmap)

        id_backmap = { coords: tid for tid, coords in enumerate(idmap) }
        object.__setattr__(self, 'id_backmap', id_backmap)

    @functools.cached_property
    def ntiles(self): return self.nrows * self.ncols

    @functools.lru_cache(maxsize=2)
    def vlen(self, dtype : Dtype): return self.vbits / 8 / Dtype.sizeof(dtype)

    @functools.lru_cache(maxsize=2)
    def peak_opc(self, dtype : Dtype):
        return self.vlen(dtype) * self.macs * 2

    def tile_coords(self, tid):
        assert tid >= 0 and tid < self.ntiles
        return self.idmap[tid]

    def coords_tile(self, r, c):
        return self.id_backmap[(r, c)]

    @functools.lru_cache(maxsize=2048)
    def addr_llc_coords(self, addr : int):
        tid = (addr >> self.l1.lbits(self.line_size)) & (self.ntiles - 1)
        return (tid // self.ncols), (tid % self.ncols)

    @property
    def defaults(self): raise NotImplementedError()

@dataclass(order=True, frozen=True)
class OracleArch(Arch):
    defaults = {
        'freq': 2.4e9,
        'vbits': 512,
        'macs': 1,
        'nrows': 32,
        'ncols': 64,
        'mapping': TileMapping.AFFINE,
        'perfect_compute': False,
        'noc_ports_per_dir': 1,
        'line_size': 32,
        'l1': CacheParams(nbanks=1, capacity=65536, assoc=16, rports=2)
    }

@dataclass(order=True, frozen=True)
class CoarseOracle(Arch):
    defaults = {
        'freq': 2.4e9,
        'vbits': 512,
        'macs': 32,
        'nrows': 8,
        'ncols': 8,
        'mapping': TileMapping.AFFINE,
        'perfect_compute': True,
        'noc_ports_per_dir': 1,
        'line_size': 32,
        'l1': CacheParams(nbanks=1, capacity=256 * 2**10, assoc=256, rports=2)
    }

@dataclass(order=True, frozen=True)
class BgroupArch(Arch):
    """Broadcast-group Architecture.

    This architecture designates subsets of tiles as broadcast groups which can
    be the target of a PUSH memory read. This represents a more realistic
    architecture where the destination bitmask of a cache line can fit within
    the packet header on the network.

    # of groups = (nrows / grows) * (ncols / gcols)
    """
    grows : int = None
    gcols : int = None

    defaults = {
        'freq': 2.4e9,
        'vbits': 512,
        'macs': 1,
        'nrows': 32,
        'ncols': 64,
        'mapping': TileMapping.AFFINE,
        'perfect_compute': False,
        'noc_ports_per_dir': 1,
        'line_size': 32,
        'l1': CacheParams(nbanks=1, capacity=65536, assoc=16, rports=2),
        'grows': 4,
        'gcols': 8
    }


@dataclass(order=True, frozen=True)
class FbcastArch(Arch):
    max_dests : int = None

    defaults = {
        'freq': 2.4e9,
        'vbits': 512,
        'macs': 1,
        'nrows': 32,
        'ncols': 64,
        'mapping': TileMapping.AFFINE,
        'perfect_compute': False,
        'noc_ports_per_dir': 1,
        'line_size': 32,
        'l1': CacheParams(nbanks=1, capacity=65536, assoc=16, rports=2),
        'max_dests': 8
    }

@dataclass(order=True, frozen=True)
class HierArch(Arch):
    grows : int = None
    gcols : int = None
    l2 : CacheParams = None

    def __post_init__(self):
        super().__post_init__()
        assert self.mapping == TileMapping.AFFINE, \
            'HierArch only works with affine mapping'

    @functools.cached_property
    def tiles_per_group(self): return self.grows * self.gcols

    @functools.cached_property
    def ngroups(self): return self.ntiles // self.tiles_per_group

    @functools.cached_property
    def groups_per_row(self): return self.nrows // self.grows

    @functools.cached_property
    def groups_per_col(self): return self.ncols // self.gcols

    @functools.lru_cache(maxsize=2048)
    def tid_to_gid(self, tid : int):
        r, c = self.tile_coords(tid)
        return r // self.grows * self.groups_per_col + c // self.gcols

    @functools.lru_cache(maxsize=2048)
    def group_base_coords(self, gid):
        assert gid >= 0 and gid < self.ngroups
        gr, gc = gid // self.groups_per_col, gid % self.groups_per_col
        return gr * self.grows, gc * self.gcols

    @functools.lru_cache(maxsize=2048)
    def addr_l2_coords(self, addr : int, gid : int):
        assert gid >= 0 and gid < self.ngroups
        br, bc = self.group_base_coords(gid)
        gtid = (addr >> self.l2.lbits(self.line_size)) & (self.tiles_per_group - 1)
        r, c = br + (gtid // self.gcols), bc + (gtid % self.gcols)
        return r, c

    defaults = {
        'freq': 2.4e9,
        'vbits': 512,
        'macs': 1,
        'nrows': 32,
        'ncols': 64,
        'mapping': TileMapping.AFFINE,
        'perfect_compute': False,
        'noc_ports_per_dir': 1,
        'line_size': 32,
        'l1': CacheParams(nbanks=None, capacity=32 * 2**10, assoc=8, rports=2),
        'grows': 4,
        'gcols': 8,
        'l2': CacheParams(nbanks=None, capacity=512 * 2**10, assoc=8, rports=1),
    }



def arch_cli_params(parser):
    parser.add_argument('-r', '--arch', type=str, default='hier')
    parser.add_argument('-g', '--geom', type=str, default=None)
    parser.add_argument('--noc-ports', type=int, default=None)
    parser.add_argument('--l1', type=str, default=None)
    parser.add_argument('--l2', type=str, default=None)
    parser.add_argument('--tiles-per-group', type=int, default=None)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--max-dests', type=int, default=None)
    parser.add_argument('--mapping', type=str, default=None)
    parser.add_argument('-x', '--perfect-compute', action='store_true')

def arch_factory(arch_name, kwargs):
    """Factory function for creating UPCYCLE architectures."""

    arch_cls = {
        'oracle': OracleArch,
        'bg': BgroupArch,
        'fbc': FbcastArch,
        'hier': HierArch,
        'coarse': CoarseOracle
    }[arch_name]

    args = arch_cls.defaults.copy()
    keys = set(args.keys())

    if kwargs['geom'] is not None: args['nrows'], args['ncols'] = map(int, kwargs['geom'].split(','))
    if kwargs['l1'] is not None: kwargs['l1'] = CacheParams.from_str(kwargs['l1'])
    if kwargs['l2'] is not None: kwargs['l2'] = CacheParams.from_str(kwargs['l2'])
    if kwargs['mapping'] is not None: kwargs['mapping'] = TileMapping.from_str(kwargs['mapping'])
    if kwargs['noc_ports'] is not None: kwargs['noc_ports_per_dir'] = kwargs['noc_ports']

    for k, v in kwargs.items():
        if v is not None and k in keys: args[k] = v

    return arch_cls(**args)

def arch_from_cli(cli_args):
    return arch_factory(cli_args.arch, cli_args.__dict__)
