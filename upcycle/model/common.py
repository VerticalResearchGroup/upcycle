from dataclasses import dataclass
from random import shuffle
from typing import Callable, Iterator
import multiprocessing
import numpy as np
import itertools
import functools
import logging
import gc

from ..common import *
from . import cache
from . import destlist
from . import noc

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=4096)
def _gen_offs(stride, d):
    if isinstance(d, int): return [d * stride]
    elif isinstance(d, Slice):
        return list(map(lambda x: x * stride, d.indices))

@functools.lru_cache(maxsize=4096)
def _gen_ids_rec(shape, strides, idx):
    if len(shape) == 1: return list(_gen_offs(strides[0], idx[0]))
    else: return sum([
        list(map(lambda i: i + off, _gen_ids_rec(shape[1:], strides[1:], idx[1:])))
        for off in _gen_offs(strides[0], idx[0])
    ], start=[])

# @functools.lru_cache
# def _gen_offs(strides, i, d):
#     if isinstance(d, int): return [d * strides[i]]
#     elif isinstance(d, Slice):
#         return list(map(lambda x: x * strides[i], d.indices))

# @functools.lru_cache
# def _gen_ids_rec(shape, strides, di, idx):
#     d = idx[di]
#     if di == len(shape) - 1: return list(_gen_offs(strides, di, d))
#     else: return sum([
#         list(map(lambda i: i + off, _gen_ids_rec(shape, strides, di + 1, idx)))
#         for off in _gen_offs(strides, di, d)
#     ], start=[])

class WorkList: ...


@dataclass(frozen=True)
class Tensor:
    arch : Arch
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

        line_mask = (1 << int(np.ceil(np.log2(self.arch.line_size)))) - 1
        object.__setattr__(self, 'line_mask', line_mask)

        assert np.prod(self.shape) * Dtype.sizeof(self.dtype) < 2**32, \
            f'Address space doesn\'t support Tensors > 4GB!'

    def _getlines(self, idx):
        assert len(self.shape) == len(idx)
        upper = self.oid << 32
        last = None
        lines = []

        for off in _gen_ids_rec(self.shape, self.strides, idx):
            line = (upper | off) & ~self.line_mask
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

def get_dests(arch : Arch, mask):
    def tid_to_rc(tid): return (tid // arch.ncols, tid % arch.ncols)
    return list(tid_to_rc(tid) for tid in mask.tiles())

class SimBase:
    def __init__(self, arch):
        self.arch = arch

    def place_work(self, tid, wl : list[WorkItem]): raise NotImplementedError()

    def flatmap_place(self, vtiles : list[list[WorkItem]], offset=0, bbox=None, randomize=False):
        if bbox is None: bbox = (0, self.arch.nrows, 0, self.arch.ncols)
        if randomize: shuffle(vtiles)
        if len(bbox) == 2: bbox = (bbox[0], bbox[0] + 1, bbox[1], bbox[1] + 1)

        bbox_nrows = bbox[1] - bbox[0]
        bbox_ncols = bbox[3] - bbox[2]
        bbox_ntiles = bbox_nrows * bbox_ncols

        for vtid, vtile in enumerate(vtiles):
            ltid = (vtid + offset) % bbox_ntiles
            lr, lc = (ltid // bbox_ncols), (ltid % bbox_ncols)
            r, c = bbox[0] + lr, bbox[2] + lc
            tid = r * self.arch.ncols + c
            self.place_work(tid, vtile)

        return len(vtiles)

def place_op(mode : str, arch : Arch, op : Operator, sim : SimBase, check_flops=True):
    global placement_funcs
    logger.debug(f'place_op(mode={mode}): {arch} {op}')
    placement_funcs[(mode, type(arch), type(op))](arch, op, sim)
    if check_flops and op.flops != sim.flops:
        logger.error(f'Placement produced different number of FLOPs! (op={op.flops} != wl={sim.flops}, wl/op={sim.flops / op.flops}x)')

class StepCounter(SimBase):
    def __init__(self, arch : Arch):
        super().__init__(arch)
        self.cur_step = [0 for _ in range(arch.ntiles)]

    def place_work(self, tid, wl : list[WorkItem]):
        self.cur_step[tid] += len(wl)

    @property
    def nsteps(self): return max(self.cur_step)

class Sim(SimBase):
    def __init__(self, arch : Arch):
        super().__init__(arch)
        self.dest_maps = {}
        self.exec_cycles = {}
        self.l1_accesses = 0
        self.l1_hits = 0
        self.flops = 0
        self.cur_step = [0 for _ in range(arch.ntiles)]
        self.l1 = [
            [cache.Cache(arch.l1_nset, arch.l1_assoc, arch.lbits) for _ in range(arch.ncols)]
            for _ in range(arch.nrows)
        ]


    def log_exec_cycles(self, step, tid, ncycles):
        if step not in self.exec_cycles:
            self.exec_cycles[step] = np.zeros(self.arch.ntiles, dtype=np.int32)
        self.exec_cycles[step][tid] += ncycles

    def log_read(self, step, tid, laddr):
        if step not in self.dest_maps:
            self.dest_maps[step] = destlist.DestList()
        self.dest_maps[step].set(laddr, tid)

    def place_work(self, tid, wl : list[WorkItem]):
        r, c = self.arch.tile_coords(tid)
        tile_cur_step = self.cur_step[tid]
        self.cur_step[tid] += len(wl)

        for i, wi in enumerate(wl):
            self.l1[r][c].reset()
            step = tile_cur_step + i

            self.log_exec_cycles(step, tid, wi.exec_lat)
            self.flops += wi.flops

            for l in wi.read_trace:
                if self.l1[r][c].lookup(l):
                    self.l1[r][c].insert(l)
                    continue
                self.l1[r][c].insert(l)
                self.log_read(step, tid, l)

            self.l1_accesses += self.l1[r][c].get_accesses()
            self.l1_hits += self.l1[r][c].get_hits()


    @property
    def nsteps(self): return max(self.cur_step)


if False:
    def simulate_tiles(arch : Arch, kwstats : dict, l1 : list[list], wl : WorkList, step : int):
        exec_cyc = []
        idle_tiles = 0
        flops = 0
        dest_map = destlist.DestList()
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
                    dest_map.set(l, r * arch.ncols + c)

                accesses += l1[r][c].get_accesses()
                hits += l1[r][c].get_hits()

        max_exec_cyc = max(exec_cyc)

        if 'max_lines' not in kwstats: kwstats['max_lines'] = 0
        kwstats['max_lines'] = max(kwstats['max_lines'], len(dest_map))

        if 'tot_lines' not in kwstats: kwstats['tot_lines'] = 0
        kwstats['tot_lines'] += len(dest_map)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'+ Max exec cycles: {max_exec_cyc}')
            logger.debug(f'+ Avg exec cycles: {np.average(exec_cyc)}')
            logger.debug(f'+ Idle tiles: {idle_tiles}')
            logger.debug(f'+ Step flops: {flops}')
            logger.debug(f'+ L1 Hit Rate: {hits} / {accesses} = {np.round(hits / max(accesses, 1), 2) * 100}%')

            # dsts = [len(d) for _, d in dest_map.items()]
            # if len(dsts) > 0:
            #     avg_dests_per_line = np.average(dsts)
            #     logger.debug(f'+ Avg dests per line: {avg_dests_per_line}')

            logger.debug(f'+ # lines transmitted: {len(dest_map)}')

        return max_exec_cyc, dest_map

def simulate_noc(arch : Arch, dest_map : dict, addr_llc_coords : Callable):
    raise NotImplementedError()

def num_steps(
    arch : Arch,
    op : Operator,
    placement_mode : str = 'naive',
    **kwargs
):
    sim = StepCounter(arch)
    place_op(placement_mode, arch, op, sim, check_flops=False)
    return sim.nsteps

def common_sim(
    arch : Arch,
    op : Operator,
    noc_sim_func : Callable = simulate_noc,
    placement_mode : str = 'naive',
    counter : multiprocessing.Value = None,
    lock : multiprocessing.Lock = None,
):
    sim = Sim(arch)
    kwstats = {}
    place_op(placement_mode, arch, op, sim)

    logger.debug(f'Simulating {sim.nsteps} steps with {sim.flops} flops...')
    traffic = noc.zero_traffic(arch, sim.nsteps)
    cycles = 0
    compute_cyc = 0


    for step in range(sim.nsteps):
        dest_map = sim.dest_maps.get(step, None)
        max_exec_cyc = max(sim.exec_cycles[step])
        traffic[step, :, :, :] = noc_sim_func(arch, kwstats, dest_map)
        net_latency = np.max(traffic[step, :, :, :]) / arch.noc_ports_per_dir

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Step {step + 1}/{sim.nsteps}: Exec latency: {compute_cyc} cyc, Noc latency: {net_latency} cyc')

        cycles += max(compute_cyc, net_latency)
        compute_cyc = max_exec_cyc

        if lock is not None and counter is not None:
            with lock: counter.value += 1

    logger.debug(f'Compute drain latency: {compute_cyc}')

    cycles += compute_cyc
    return SimResult(sim.nsteps, cycles, traffic, kwstats)
