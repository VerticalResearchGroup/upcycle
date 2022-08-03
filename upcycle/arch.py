from dataclasses import dataclass, field
from enum import IntEnum
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

@dataclass(order=True, frozen=True)
class Arch:
    freq : float
    vbits : int
    macs : int
    nrows : int
    ncols : int
    mapping : TileMapping = TileMapping.AFFINE
    perfect_compute : bool = False
    noc_ports_per_dir : int = 1
    line_size : int = 64
    l1_capacity : int = 16384
    l1_assoc : int = 16
    l1_rports : int = 2

    @property
    def vbytes(self): return self.vbits // 8

    @property
    def l1_nset(self):
        return int(self.l1_capacity / self.line_size / self.l1_assoc)

    @property
    def lbits(self): return int(np.ceil(np.log2(self.line_size)))

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
        tid = (addr >> self.lbits) & (self.ntiles - 1)
        return (tid // self.ncols), (tid % self.ncols)


@dataclass(order=True, frozen=True)
class BgroupArch(Arch):
    grows : int = 4
    gcols : int = 8

@dataclass(order=True, frozen=True)
class OracleArch(Arch): pass

@dataclass(order=True, frozen=True)
class FbcastArch(Arch):
    max_dests : int = 8

def arch_cli_params(parser):
    parser.add_argument('-r', '--arch', type=str, default='oracle')
    parser.add_argument('--noc-ports', type=int, default=1)
    parser.add_argument('--l1-capacity', type=int, default=64*1024)
    parser.add_argument('--l1-assoc', type=int, default=16)
    parser.add_argument('--l1-rports', type=int, default=2)
    parser.add_argument('--line-size', type=int, default=32)
    parser.add_argument('--group', type=str, default='4,8')
    parser.add_argument('--max-dests', type=int, default=8)
    parser.add_argument('--mapping', type=str, default='affine')
    parser.add_argument('-x', '--perfect-compute', action='store_true')

def arch_factory(arch_name, freq=2.4e9, vbits=512, macs=1, nrows=32, ncols=64, **kwargs):
    mapping = TileMapping.AFFINE
    if kwargs['mapping'] == 'hilbert':
        mapping = TileMapping.HILBERT

    if arch_name == 'oracle':
        arch = OracleArch(
            freq, vbits, macs, nrows, ncols, mapping, kwargs['perfect_compute'],
            kwargs['noc_ports'], kwargs['line_size'], kwargs['l1_capacity'], kwargs['l1_assoc'], kwargs['l1_rports'])
    elif arch_name == 'bg':
        [grows, gcols] = list(map(int, kwargs['group'].split(',')))
        arch = BgroupArch(
            freq, vbits, macs, nrows, ncols, mapping, kwargs['perfect_compute'],
            kwargs['noc_ports'], kwargs['line_size'], kwargs['l1_capacity'], kwargs['l1_assoc'], kwargs['l1_rports'],
            grows, gcols)
    elif arch_name == 'fbc':
        arch = FbcastArch(
            freq, vbits, macs, nrows, ncols, mapping, kwargs['perfect_compute'],
            kwargs['noc_ports'], kwargs['line_size'], kwargs['l1_capacity'], kwargs['l1_assoc'], kwargs['l1_rports'],
            kwargs['max_dests'])

    return arch

def arch_from_cli(cli_args):
    return arch_factory(cli_args.arch, **cli_args.__dict__)
