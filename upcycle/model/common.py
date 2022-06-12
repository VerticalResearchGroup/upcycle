from dataclasses import dataclass
from random import shuffle
from typing import Iterator
import numpy as np
import itertools
import functools
import logging

from ..common import *

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Tensor:
    oid : int
    dtype : Dtype
    shape : tuple
    strides : tuple = None

    def __post_init__(self):
        object.__setattr__(
            self,
            'strides',
            tuple(
                int(np.prod(self.shape[i+1:]) * Dtype.sizeof(self.dtype))
                for i in range(len(self.shape))))

        object.__setattr__(self, 'linecache', {})

        assert np.prod(self.shape) * Dtype.sizeof(self.dtype) < 2**32, \
            f'Address space doesn\'t support Tensors > 4GB!'

    def _gen_offs(self, i, d):
        if isinstance(d, int):
            yield d * self.strides[i]
        elif isinstance(d, Slice):
            yield from map(lambda x: x * self.strides[i], d.indices)

    def _gen_ids_rec(self, di, idx):
        d = idx[di]
        if di == len(self.shape) - 1:
            yield from self._gen_offs(di, d)
        else:
            for off in self._gen_offs(di, d):
                yield from map(lambda i: i + off, self._gen_ids_rec(di + 1, idx))

    @functools.lru_cache
    def _getlines(self, idx):
        assert len(self.shape) == len(idx)
        upper = self.oid << 32
        last = None
        lines = []

        for off in self._gen_ids_rec(0, idx):
            line = (upper | off) & ~0x3F
            if last is not None and line == last: continue
            last = line
            lines.append(line)

        return lines

    def __getitem__(self, _idx):
        def _convert_slice(x):
            (i, s) = x
            if isinstance(s, int): return s
            elif isinstance(s, Slice): return s
            elif isinstance(s, slice):
                return Slice.from_pyslice(s, self.shape[i])
            else: assert False, f'Invalid slice: {s}'
        idx = tuple(map(_convert_slice, enumerate(_idx)))
        return self._getlines(idx)


@dataclass(frozen=True)
class WorkItem:
    arch : Arch
    dtype : Dtype

    @property
    def exec_lat(self): raise NotImplementedError()

    @property
    def flops(self): raise NotImplementedError()

    @property
    def read_trace(self)  -> Iterator[int]: raise NotImplementedError()

    @property
    def write_trace(self)  -> Iterator[int]: raise NotImplementedError()


@dataclass(frozen=True)
class WorkItemPerfectCompute(WorkItem):
    @property
    def exec_lat(self): return self.flops / self.arch.peak_opc(self.dtype)


@dataclass
class WorkList:
    arch : Arch
    tensors : list[Tensor]
    tiles : list[list[list[WorkItem]]]

    @property
    def flattiles(self):
        for row in self.tiles:
            for tile in row:
                yield tile

    @property
    def workitems(self):
        for row in self.tiles:
            for tile in row:
                yield from tile

    @staticmethod
    def from_arch(arch : Arch, tensors : list[Tensor]):
        return WorkList(
            arch,
            tensors,
            [[list() for _ in range(arch.ncols)] for _ in range(arch.nrows)])


    def contract1d_place(self, row, vtiles : list[list[WorkItem]]):
        wi_per_col = len(vtiles) // self.arch.nrows
        off = 0

        for col in range(self.arch.ncols):
            ntiles = wi_per_col
            if col < (len(vtiles) % self.arch.ncols): ntiles += 1
            tl = self[row, col]
            tl += list(itertools.chain(*vtiles[off : off + ntiles]))
            off += ntiles


    def contract2d_place(self, vtiles : list[list[list[WorkItem]]]):
        wi_per_row = len(vtiles) // self.arch.nrows
        off = 0

        for row in range(self.arch.nrows):
            ntiles = wi_per_row
            if row < (len(vtiles) % self.arch.nrows): ntiles += 1
            for row_i in range(off, off + ntiles):
                self.contract1d_place(row, vtiles[row_i])
            off += ntiles

    def flatmap_place(self, vtiles : list[list[WorkItem]], bbox=None, randomize=False):
        if bbox is None: bbox = (0, self.arch.nrows, 0, self.arch.ncols)
        if randomize: shuffle(vtiles)

        bbox_nrows = bbox[1] - bbox[0]
        bbox_ncols = bbox[3] - bbox[2]
        bbox_ntiles = bbox_nrows * bbox_ncols

        for vtid, vtile in enumerate(vtiles):
            tid = vtid % bbox_ntiles
            lr, lc = (tid // bbox_ncols), (tid % bbox_ncols)
            tl = self[bbox[0] + lr, bbox[2] + lc]
            tl += list(itertools.chain(vtile))


    @property
    def nsteps(self): return max(map(len, self.flattiles))

    @property
    def flops(self):
        return sum(map(lambda x: x.flops, self.workitems))

    def __getitem__(self, idx):
        r, c = idx
        return self.tiles[r][c]

placement_funcs = {}

def register_placement(mode, archclass, opclass):
    def decorator(x):
        placement_funcs[(mode, archclass, opclass)] = x
        return x
    return decorator

def place_op(mode : str, arch : Arch, op : Operator) -> WorkList:
    global placement_funcs
    logger.debug(f'place_op(mode={mode}): {arch} {op}')
    wl : WorkList = placement_funcs[(mode, type(arch), type(op))](arch, op)
    assert op.flops == wl.flops, f'{op.flops} != {wl.flops}, {wl.flops / op.flops}'
    return wl

@dataclass(frozen=True)
class SimResult:
    nsteps : int
    cycles : int
    traffic : np.ndarray

sim_funcs = {}

def register_sim(archclass):
    def decorator(func):
        global sim_funcs
        sim_funcs[archclass] = func
        return func
    return decorator

def simulate(arch : Arch, op : Operator, *args, **kwargs) -> SimResult:
    global sim_funcs
    return sim_funcs[type(arch)](arch, op, *args, **kwargs)
