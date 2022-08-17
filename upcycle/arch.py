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

    @property
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

    @property
    def ntiles(self): return self.nrows * self.ncols

    def vlen(self, dtype : Dtype): return self.vbits / 8 / Dtype.sizeof(dtype)

    def peak_opc(self, dtype : Dtype):
        return self.vlen(dtype) * self.macs * 2

    def tile_coords(self, tid):
        assert tid >= 0 and tid < self.ntiles
        return self.idmap[tid]

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
    tiles_per_group : int = None
    l2 : CacheParams = None

    @property
    def ntiles(self): return self.nrows * self.ncols * self.tiles_per_group

    @property
    def ngroups(self): return self.nrows * self.ncols

    def tid_to_gid(self, tid):
        return tid // self.tiles_per_group

    def tile_coords(self, tid):
        assert tid >= 0 and tid < self.ntiles
        gid = self.tid_to_gid(tid)
        r, c = self.idmap[gid]
        return r, c

    def group_coords(self, gid):
        assert gid >= 0 and gid < self.ngroups
        r, c = self.idmap[gid]
        return r, c

    def addr_llc_coords(self, addr : int):
        gid = (addr >> self.l1.lbits(self.line_size)) & (self.ngroups - 1)
        return (gid // self.ncols), (gid % self.ncols)

    defaults = {
        'freq': 2.4e9,
        'vbits': 512,
        'macs': 1,
        'nrows': 4,
        'ncols': 16,
        'mapping': TileMapping.AFFINE,
        'perfect_compute': False,
        'noc_ports_per_dir': 1,
        'line_size': 64,
        'l1': CacheParams(nbanks=None, capacity=64 * 2**10, assoc=16, rports=2),
        'tiles_per_group': 32,
        'l2': CacheParams(nbanks=None, capacity=256 * 2**10, assoc=8, rports=1),
    }



def arch_cli_params(parser):
    parser.add_argument('-r', '--arch', type=str, default='oracle')
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
        'hier': HierArch
    }[arch_name]

    args = arch_cls.defaults.copy()
    keys = set(args.keys())

    if kwargs['l1'] is not None: args['l1'] = CacheParams.from_str(kwargs['l1'])
    if kwargs['l2'] is not None: args['l2'] = CacheParams.from_str(kwargs['l2'])
    if kwargs['mapping'] is not None: args['mapping'] = TileMapping.from_str(v)

    for k, v in kwargs.items():
        if v is not None and k in keys: args[k] = v

    return arch_cls(**args)

def arch_from_cli(cli_args):
    return arch_factory(cli_args.arch, cli_args.__dict__)
