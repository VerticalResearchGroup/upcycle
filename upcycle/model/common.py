from dataclasses import dataclass
from random import shuffle
from typing import Callable, Iterator
import numpy as np
import itertools
import functools
import logging

from ..common import *
from . import cache
from . import noc

logger = logging.getLogger(__name__)

@functools.lru_cache
def _gen_offs(strides, i, d):
    if isinstance(d, int): return [d * strides[i]]
    elif isinstance(d, Slice):
        return list(map(lambda x: x * strides[i], d.indices))

@functools.lru_cache
def _gen_ids_rec(shape, strides, di, idx):
    d = idx[di]
    if di == len(shape) - 1: return list(_gen_offs(strides, di, d))
    else: return sum([
        list(map(lambda i: i + off, _gen_ids_rec(shape, strides, di + 1, idx)))
        for off in _gen_offs(strides, di, d)
    ], start=[])


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

    @functools.lru_cache
    def _getlines(self, idx):
        assert len(self.shape) == len(idx)
        upper = self.oid << 32
        last = None
        lines = []

        for off in _gen_ids_rec(self.shape, self.strides, 0, idx):
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

def register_placement_single(mode, archclass, opclass, f):
    placement_funcs[(mode, archclass, opclass)] = f

def register_placement(mode_s, archclass_s, opclass_s):
    if not isinstance(mode_s, list):
        mode_s = [mode_s]
    if not isinstance(archclass_s, list):
        archclass_s = [archclass_s]
    if not isinstance(opclass_s, list):
        opclass_s = [opclass_s]

    def decorator(f):
        for mode in mode_s:
            for archclass in archclass_s:
                for opclass in opclass_s:
                    register_placement_single(mode, archclass, opclass, f)
        return f
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
    kwstats : dict

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

def simulate_tiles(arch : Arch, kwstats : dict, l1 : list[list], wl : WorkList, step : int):
    exec_cyc = []
    idle_tiles = 0
    flops = 0
    dest_map = dict()
    accesses = 0
    hits = 0

    for r in range(arch.nrows):
        for c in range(arch.ncols):
            tile = wl[r, c]
            l1[r][c].reset()
            if step >= len(tile):
                idle_tiles += 1
                continue
            exec_cyc.append(tile[step].exec_lat)
            flops += tile[step].flops
            for l in tile[step].read_trace:
                if l1[r][c].lookup(l):
                    l1[r][c].insert(l)
                    continue
                l1[r][c].insert(l)
                if l not in dest_map: dest_map[l] = []
                dest_map[l].append((r, c))

            accesses += l1[r][c].get_accesses()
            hits += l1[r][c].get_hits()

    max_exec_cyc = max(exec_cyc)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Max exec cycles: {max_exec_cyc}')
        logger.debug(f'+ Avg exec cycles: {np.average(exec_cyc)}')
        logger.debug(f'+ Idle tiles: {idle_tiles}')
        logger.debug(f'+ Step flops: {flops}')
        logger.debug(f'+ L1 Hit Rate: {hits} / {accesses} = {np.round(hits / max(accesses, 1), 2) * 100}%')

        dsts = [len(d) for _, d in dest_map.items()]
        if len(dsts) > 0:
            avg_dests_per_line = np.average(dsts)
            logger.debug(f'+ Avg dests per line: {avg_dests_per_line}')

        logger.debug(f'+ # lines transmitted: {len(dest_map)}')

    return max_exec_cyc, dest_map

def simulate_noc(arch : Arch, dest_map : dict, addr_llc_coords : Callable):
    raise NotImplementedError()

def common_sim(
    arch : Arch,
    op : Operator,
    tile_sim_func : Callable = simulate_tiles,
    noc_sim_func : Callable = simulate_noc,
    placement_mode : str = 'naive',
    l1_capacity : int = 16384,
    l1_assoc : int = 4,
    randomize_llc : bool = False,
):
    cycles = 0
    kwstats = dict()

    if randomize_llc:
        llc_addr_map = list(range(arch.ntiles))
        random.shuffle(llc_addr_map)
    else:
        llc_addr_map = None

    def tile_coords(tid):
        assert tid >= 0 and tid < arch.ntiles
        return (tid // arch.ncols), (tid % arch.ncols)

    def addr_llc_coords(addr : int):
        line = addr >> 6
        tid = line & (arch.ntiles - 1)
        if llc_addr_map is not None:
            return tile_coords(llc_addr_map[tid])
        else:
            return tile_coords(tid)


    wl = place_op(placement_mode, arch, op)
    l1_nway = int(l1_assoc)
    l1_nset = int(l1_capacity / 64 / l1_nway)
    l1 = [
        [cache.Cache(l1_nset, l1_nway, 6) for _ in range(arch.ncols)]
        for _ in range(arch.nrows)
    ]

    compute_cyc = 0

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'Simulating {wl.nsteps} steps with {wl.flops} flops...')

    traffic = noc.zero_traffic(arch, wl.nsteps)

    for step in range(wl.nsteps):
        logger.debug(f'Step {step}')

        max_exec_cyc, dest_map = tile_sim_func(arch, kwstats, l1, wl, step)
        traffic[step, :, :, :] = noc_sim_func(arch, kwstats, dest_map, addr_llc_coords)

        net_latency = np.max(traffic[step, :, :, :]) / arch.noc_ports_per_dir

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'+ Exec latency: {compute_cyc} cyc')
            logger.debug(f'+ Noc latency: {net_latency} cyc')

        cycles += max(compute_cyc, net_latency)
        compute_cyc = max_exec_cyc

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Compute drain latency: {compute_cyc}')

    cycles += compute_cyc

    return SimResult(wl.nsteps, cycles, traffic, kwstats)
