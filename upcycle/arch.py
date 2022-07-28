from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
import random
import logging

from .common import *

logger = logging.getLogger(__name__)

@dataclass(order=True, frozen=True)
class Arch:
    freq : float
    vbits : int
    macs : int
    nrows : int
    ncols : int
    noc_ports_per_dir : int = 1
    line_size : int = 64
    l1_capacity : int = 16384
    l1_assoc : int = 16

    @property
    def l1_nset(self):
        return int(self.l1_capacity / self.line_size / self.l1_assoc)

    @property
    def lbits(self): return int(np.ceil(np.log2(self.line_size)))

    def __post_init__(self):
        if self.noc_ports_per_dir > 1:
            logger.warn(f'Arch has {self.noc_ports_per_dir} ports per direction (>1)')

    @property
    def ntiles(self): return self.nrows * self.ncols

    def vlen(self, dtype : Dtype): return self.vbits / 8 / Dtype.sizeof(dtype)

    def peak_opc(self, dtype : Dtype):
        return self.vlen(dtype) * self.macs * 2

    def tile_coords(self, tid):
        assert tid >= 0 and tid < self.ntiles
        return (tid // self.ncols), (tid % self.ncols)

    def addr_llc_coords(self, addr : int):
        line = addr >> self.lbits
        return self.tile_coords(line & (self.ntiles - 1))


@dataclass(order=True, frozen=True)
class BgroupArch(Arch):
    grows : int = 4
    gcols : int = 8

@dataclass(order=True, frozen=True)
class OracleArch(Arch): pass

def arch_cli_params(parser):
    parser.add_argument('-r', '--arch', type=str, default='oracle')
    parser.add_argument('--noc-ports', type=int, default=1)
    parser.add_argument('--l1-capacity', type=int, default=64*1024)
    parser.add_argument('--l1-assoc', type=int, default=16)
    parser.add_argument('--line-size', type=int, default=32)
    parser.add_argument('--group', type=str, default='4,8')

def arch_factory(arch_name, freq=2.4e9, vbits=512, macs=1, nrows=32, ncols=64, **kwargs):
    if arch_name == 'oracle':
        arch = OracleArch(
            freq, vbits, macs, nrows, ncols,
            kwargs['noc_ports'], kwargs['line_size'], kwargs['l1_capacity'], kwargs['l1_assoc'])
    elif arch_name == 'bg':
        [grows, gcols] = list(map(int, kwargs['group'].split(',')))
        arch = BgroupArch(
            freq, vbits, macs, nrows, ncols,
            kwargs['noc_ports'], kwargs['line_size'], kwargs['l1_capacity'], kwargs['l1_assoc'],
            grows, gcols)

    return arch

def arch_from_cli(cli_args):
    return arch_factory(cli_args.arch, **cli_args.__dict__)
