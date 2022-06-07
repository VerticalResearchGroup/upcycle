from dataclasses import dataclass
from random import shuffle
from typing import Iterator
import numpy as np
import itertools
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

    def _gen_offs(self, i, d):
        if isinstance(d, int):
            yield d * self.strides[i]
        elif isinstance(d, slice):
            start = 0 if d.start is None else d.start
            stop = self.shape[i] if d.stop is None else d.stop
            yield from map(
                lambda x: x * self.strides[i], range(start, stop, d.step))


    def _gen_ids_rec(self, di, idx):
        d = idx[di]
        if di == len(self.shape) - 1:
            yield from self._gen_offs(di, d)
        else:
            for off in self._gen_offs(di, d):
                yield from map(
                    lambda i: i + off, self._gen_ids_rec(di + 1, idx))

    @staticmethod
    def hashidx(idx):
        def h(x):
            if isinstance(x, slice): return hash((x.start, x.stop, x.step))
            else: return hash(x)
        return hash(tuple(h(x) for x in idx))

    def __getitem__(self, idx):
        assert len(self.shape) == len(idx)
        upper = self.oid << 32
        last = None
        lines = []

        key = self.hashidx(idx) % 4096
        if key in self.linecache and self.linecache[key][0] == idx:
            return self.linecache[key][1]

        for off in self._gen_ids_rec(0, idx):
            line = (upper | off) & ~0x3F
            if last is not None and line == last: continue
            last = line
            # yield line
            lines.append(line)

        self.linecache[key] = (idx, lines)

        return lines


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

class Soc:
    def __init__(self, arch : Arch, placement_mode : str = 'naive'):
        self.arch = arch
        self.placement_mode = placement_mode

    def simulate(self, op : Operator): raise NotImplementedError()

    def noc(self, step : int): raise NotImplementedError()

    @property
    def nsteps(self): raise NotImplementedError()

    @property
    def cycles(self): raise NotImplementedError()

    @property
    def total_hops(self):
        return sum(self.noc(step).total_hops for step in range(self.nsteps))


socs = {}

def register_soc(archclass):
    def decorator(xclass):
        global socs
        socs[archclass] = xclass
        return xclass
    return decorator

def make_soc(arch : Arch, *args, **kwargs) -> Soc:
    global socs
    return socs[type(arch)](arch, *args, **kwargs)
