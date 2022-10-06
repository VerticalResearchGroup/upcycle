from dataclasses import dataclass, fields
from random import shuffle
from typing import Callable, Generator, Iterator
import multiprocessing
import numpy as np
import itertools
import functools
import logging
import gc

from ..common import *
from ..arch import *
from . import c_model
from . import noc

logger = logging.getLogger(__name__)

def nloads(arch : Arch, dtype : Dtype, r : Slice, R : int, c : Slice, C : int, transpose=False, contig=True):
    """Compute number of loads for a given 2D slice of memory.

    This will compute the number of cache-line sized loads needed to bring a
    2D slice of memory into the core. When the rows are contiguous in memory and
    are smaller than the line-size, we can optimize loads by making use of the
    overlap.
    """
    dtsize = Dtype.sizeof(dtype)

    if transpose:
        r, c = c, r
        R, C = C, R

    if len(c) == C and contig:
        # In this case, the rows we are loading are contiguous in memory, so we
        # can extract load-reuse in the case where the length of the row is
        # smaller than the cache line size.
        return cld(len(r) * len(c) * dtsize, arch.vbytes)
    else:
        # Here, rows aren't contiguous. While there is potential to still
        # extract some reuse, we ignore that since it's not likely to be the
        # common case and won't buy us much performance.
        return len(r) * cld(len(c) * dtsize, arch.vbytes)


@functools.lru_cache(maxsize=4096)
def _gen_offs(stride, d):
    if isinstance(d, int): return [d * stride]
    elif isinstance(d, Slice):
        return list(map(lambda x: x * stride, d.indices))

@functools.lru_cache(maxsize=4096)
def _gen_ids_rec(shape, strides, idx):
    """Generate 1D offsets for a Nd-array slice.

    This function is used for producing addresses into a Tensor for a given
    slice. This is a very performance critical function so we make use of
    caching to try to speed up the process.
    """
    if len(shape) == 1: return list(_gen_offs(strides[0], idx[0]))
    else: return sum([
        list(map(lambda i: i + off, _gen_ids_rec(shape[1:], strides[1:], idx[1:])))
        for off in _gen_offs(strides[0], idx[0])
    ], start=[])

USE_C_TRACE = True

@dataclass(frozen=True)
class Tensor:
    """Represents an operand tensor for a DL operator."""
    arch : Arch
    oid : int # "Output ID"
    dtype : Dtype
    shape : tuple
    strides : tuple = None

    def __repr__(self):
        return f'Tensor[{self.oid}][{self.dtype}]({self.shape})'

    def __post_init__(self):
        # N.B. frozen dataclasses will yell at us if we assign attributes willy-
        # nilly. We use object.__setattr__ to bypass that. The strides are pre-
        # computed in the constructor for performance reasons.
        object.__setattr__(
            self,
            'strides',
            tuple(
                int(np.prod(self.shape[i+1:]) * Dtype.sizeof(self.dtype))
                for i in range(len(self.shape))))

        line_mask = (1 << int(np.ceil(np.log2(self.arch.line_size)))) - 1
        object.__setattr__(self, 'line_mask', line_mask)

        # N.B. We don't support >4GB tensors because we shift oid by 32 bits to
        # produce an address. This is a somewhat arbitrary limit so we can
        # change it in the future if needed.
        assert np.prod(self.shape) * Dtype.sizeof(self.dtype) < 2**32, \
            f'Address space doesn\'t support Tensors > 4GB!'

    def _getlines(self, idx):
        """Given a index slice, produce a line-level address trace."""
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

    def py_trace(self, _idx):
        """Frontend for _getlines."""
        def _convert_slice(x):
            (i, s) = x
            if isinstance(s, int): return s
            elif isinstance(s, Slice): return s
            elif isinstance(s, slice):
                return Slice.from_pyslice(s, self.shape[i])
            else: assert False, f'Invalid slice: {s}'
        idx = tuple(map(_convert_slice, enumerate(_idx)))
        return self._getlines(idx)

    def _convert_slice(self, i, s):
        if isinstance(s, int): return [s, s + 1, 1]
        elif isinstance(s, Slice): return [s.start, s.stop, s.step]
        elif isinstance(s, slice):
            return [
                s.start if s.start is not None else 0,
                s.stop if s.stop is not None else self.shape[i],
                s.step if s.step is not None else 1]
        else: assert False, f'Invalid slice: {s}'

    def c_trace(self, _idx):
        """Alternate C++ implementation for py_trace.

        N.B. py_trace still exists because I needed it for debugging the C++
        version. Also, the C++ version only supports upto 5D tensors, so we
        would need to fallback to the Python version if we ever run into >5D.
        """
        idx = sum((self._convert_slice(i, x) for i, x in enumerate(_idx)), start=[])
        return [c_model.AffineTile(self.oid, self.shape, self.strides, idx)]

    def __getitem__(self, _idx):
        assert len(_idx) == len(self.shape)
        # N.B. We use Python's __getitem__ as a frontend for all this indexing
        # nonsense for cosmetic reasons. This way we can make a tensor and then
        # write something like A[1:2, 5:, :] and it will produce all the
        # addresses for that slice.
        global USE_C_TRACE
        if USE_C_TRACE: return self.c_trace(_idx)
        else: return self.py_trace(_idx)

    @functools.cached_property
    def _num_elems(self): return np.prod(self.shape)

    def __len__(self): return self._num_elems

@dataclass(frozen=True)
class WorkItem:
    """Base class for all workitems."""
    arch : Arch
    op : Operator
    inputs : list[Tensor]
    outputs : list[Tensor]

    @property
    def flops(self): raise NotImplementedError()

    @property
    def perfect_exec_lat(self): return self.flops / self.arch.peak_opc(self.op.dtype)

    exec_warn = set()

    @property
    def exec_lat(self):
        if type(self) not in self.exec_warn:
            logger.warn(f'exec_lat not implemented for {type(self)}. Falling back to perfect_exec_lat.')
            self.exec_warn.add(type(self))

        return self.perfect_exec_lat

    @property
    def read_trace(self)  -> Iterator[int]: raise NotImplementedError()

    @property
    def write_trace(self)  -> Iterator[int]: raise NotImplementedError()

#
# Placement Functions.
#
# Here we create a dispatch table for operator placement. It is key'd by
# (mode, type(arch), type(op)) where,
#
# mode: the placement "mode" (arbitrary string identifier given by us)
# type(arch): the type of the architecture (e.g. "OracleArch")
# type(op): the type of the operator (e.g. "Conv2D")
#
# register_placement serves as a decorator that other parts of the codebase can
# use to insert placement functions into this table.
#

placement_funcs = {}

def register_placement_single(archclass, opclass_or_spec, f):
    if type(opclass_or_spec) is type:
        opclass = opclass_or_spec
        opspec = None
    else:
        opclass = type(opclass_or_spec)
        opspec = opclass_or_spec

    if (archclass, opclass) not in placement_funcs:
        placement_funcs[(archclass, opclass)] = []
    placement_funcs[(archclass, opclass)].append((opspec, f))

def register_placement(archclass_s, opclass_or_specs):
    if not isinstance(archclass_s, list):
        archclass_s = [archclass_s]
    if not isinstance(opclass_or_specs, list):
        opclass_or_specs = [opclass_or_specs]

    def decorator(f):
        for archclass in archclass_s:
            for opclass_or_spec in opclass_or_specs:
                register_placement_single(archclass, opclass_or_spec, f)
        return f
    return decorator

def choose_placement_func(arch : Arch, op : Operator):
    global placement_funcs
    key = (type(arch), type(op))
    if key not in placement_funcs:
        raise KeyError(f'No placement funcs for {key}')

    best_score, best_func = -1, None

    logger.debug(f'Looking for placement profile for {op}')
    for prof, f in placement_funcs[key]:
        if prof is None: valid, score = True, 0
        else: valid, score = prof.match(op)

        logger.debug(f'    + {prof} valid={valid} score={score} (best={best_score})')

        if valid and (score > best_score):
            best_score = score
            best_func = f

    if best_func is None:
        raise KeyError(f'No valid placement func for {key}')

    return best_func, best_score

def place_op(arch : Arch, op : Operator, sim, check_flops=True):
    logger.debug(f'place_op: {arch} {op}')
    func, score = choose_placement_func(arch, op)
    logger.debug(f'Chosen placement function: {op} -> {func.__name__} (score = {score})')
    func(arch, op, sim)

    if check_flops and op.flops != sim.flops:
        if np.abs(sim.flops / op.flops - 1) < 0.05:
            logger.warn(f'Placement produced different number of FLOPs! (op={op.flops} != wl={sim.flops}, wl/op={sim.flops / op.flops}x)')
            logger.warn(f'Offending op: {op}')
        else:
            logger.error(f'Placement produced different number of FLOPs! (op={op.flops} != wl={sim.flops}, wl/op={sim.flops / op.flops}x)')
            logger.error(f'Offending op: {op}')
            logger.error(f'Placement function: {func.__name__}')
            logger.error(f'Arch: {arch}')


@dataclass(frozen=True)
class SimResult:
    """The output of the simulator."""
    nsteps : int
    cycles : int
    traffic : np.ndarray
    kwstats : dict

#
# Simulator functions.
#
# Here we create a dispatch table for simulator functions. It is key'd by only
# the class of architecture being simulated (e.g. "OracleArch").
#
sim_funcs = {}

def register_sim(archclass):
    def decorator(func):
        global sim_funcs
        sim_funcs[archclass] = func
        return func
    return decorator

def simulate(arch : Arch, op : Operator, *args, **kwargs) -> SimResult:
    """Main entry point for the performance model."""
    global sim_funcs
    return sim_funcs[type(arch)](arch, op, *args, **kwargs)

def get_dest_tids(arch : Arch, mask):
    return list(tid for tid in mask.tiles())

def get_dests(arch : Arch, mask):
    def tid_to_rc(tid): return (tid // arch.ncols, tid % arch.ncols)
    return list(tid_to_rc(tid) for tid in mask.tiles())

class SimBase:
    """Base class for the simulator object.

    This class serves as the exposed interface for the placement functions. That
    is, placement gets a sim object and can use the functions declared here to
    carry out placement.

    N.B. The main reason this class is abstract is becasue I wanted a quick way
    to be able to count the number of steps needed for a given layer
    (See StepCounter below).
    """
    def __init__(self, arch : Arch):
        self.arch = arch
        self.kwstats = {}

    def place_work(self, tid, wl : WorkItem): raise NotImplementedError()

    def step(self): raise NotImplementedError()

    def drain(self): raise NotImplementedError()

    def flatmap_place(self, vtiles : list[Iterator[WorkItem]], offset=0, bbox=None, randomize=False):
        if randomize: shuffle(vtiles)
        logger.debug(f'flatmap_place: {len(vtiles)} vtiles')
        if bbox is not None: raise DeprecationWarning('bbox is deprecated')
        if offset != 0: raise DeprecationWarning('offset is deprecated')
        tiles = [None for _ in range(self.arch.ntiles)]

        logger.debug(f'{sum(blkdiv(len(vtiles), self.arch.ntiles))}')

        i = 0
        for tid, n in enumerate(blkdiv(len(vtiles), self.arch.ntiles)):
            tiles[tid] = itertools.chain(*[vtiles[i + j] for j in range(n)])
            i += n

        while True:
            placed = 0
            for tid in range(self.arch.ntiles):
                if tiles[tid] is None: continue
                try:
                    self.place_work(tid, next(tiles[tid]))
                    placed += 1
                except StopIteration:
                    tiles[tid] = None

            if placed == 0: break
            self.step()


    def map2d_place(self, vtiles : list[list[Iterator[WorkItem]]]):
        trows, tcols = len(vtiles), len(vtiles[0])
        logging.debug(f'map2d_place: {trows}x{tcols} vtiles')
        tiles = [[None for _ in range(self.arch.ncols)] for _ in range(self.arch.nrows)]

        r = 0
        for m in blkdiv(trows, self.arch.nrows):
            tiles.append([])
            c = 0
            for n in blkdiv(tcols, self.arch.ncols):
                tiles[r][c] = itertools.chain(*[
                    vtiles[r + i][c + j]
                    for i in range(m)
                    for j in range(n)])

                c += 1
            r += 1

        while True:
            placed = 0
            for r in range(self.arch.nrows):
                for c in range(self.arch.ncols):
                    if tiles[r][c] is None: continue
                    tid = r * self.arch.ncols + c
                    try:
                        self.place_work(tid, next(tiles[r][c]))
                        placed += 1
                    except StopIteration:
                        tiles[r][c] = None

            if placed == 0: break
            self.step()

    def barrier(self): raise NotImplementedError()

class StepCounter(SimBase):
    """Dummy sim object used to count number of steps for a layer.

    N.B. This is used to predict runtime ahead of execution. See num_steps()
    below.
    """
    def __init__(self, arch : Arch, **kwargs):
        super().__init__(arch)
        self.cur_step = [0 for _ in range(arch.ntiles)]
        self.flops = 0

    def place_work(self, tid, wi : WorkItem):
        # self.flops += wi.flops
        self.cur_step[tid] += 1

    def step(self): pass

    def drain(self): pass

    def barrier(self):
        max_step = max(self.cur_step)
        for tid in range(len(self.cur_step)): self.cur_step[tid] = max_step

    @property
    def nsteps(self): return max(self.cur_step)

class Sim(SimBase):
    def __init__(
        self,
        arch : Arch,
        noc_sim_func : Callable,
        total_steps : int,
        lock : any = None,
        counter : any = None,
        cancel : any = None,
    ):
        super().__init__(arch)
        self.noc_sim_func = noc_sim_func
        self.total_steps = total_steps
        self.lock = lock
        self.counter = counter
        self.cancel = cancel
        self.dest_maps = {}
        self.exec_cycles = {}
        self.perfect_exec_cycles = {}
        self.rss = [[] for _ in range(arch.ntiles)]
        self.flops = 0
        self.global_step = 0
        self.cur_step = [0 for _ in range(arch.ntiles)]
        self.l1 = [
            c_model.Cache(
                arch.l1.nset(arch.line_size),
                arch.l1.assoc,
                arch.l1.lbits(arch.line_size))
            for _ in range(arch.ntiles)
        ]
        self.total_traffic = noc.zero_traffic(arch)
        self.cycles = [0 for _ in self.arch.scales]
        self.compute_cyc = [0 for _ in self.arch.scales]
        self.kwstats['rss'] = []
        self.kwstats['l1_accesses'] = 0
        self.kwstats['l1_hits'] = 0
        self.kwstats['llc_accesses'] = 0

    def log_exec_cycles(self, step, tid, ncycles, perfect):
        if step not in self.exec_cycles:
            self.exec_cycles[step] = np.zeros(self.arch.ntiles, dtype=np.int32)
            self.perfect_exec_cycles[step] = np.zeros(self.arch.ntiles, dtype=np.int32)
        self.exec_cycles[step][tid] += ncycles
        self.perfect_exec_cycles[step][tid] += perfect

    def log_read(self, step, tid, laddr):
        if step not in self.dest_maps:
            self.dest_maps[step] = c_model.DestList()
        self.dest_maps[step].set(laddr, tid)

    def place_work(self, tid, wi : WorkItem):
        if self.cancel is not None and self.cancel.value: raise KeyboardInterrupt()
        assert isinstance(wi, WorkItem)
        global USE_C_TRACE
        self.l1[tid].reset()
        step = self.cur_step[tid]

        # if logger.isEnabledFor(logging.DEBUG) and tid == 0:
        #     logger.debug(f'Tile 0: step={step} wi={wi}')

        if step not in self.dest_maps:
            self.dest_maps[step] = c_model.DestList()

        # if logger.isEnabledFor(logging.DEBUG):
        if wi.perfect_exec_lat > wi.exec_lat:
            logger.error(f'Perfect exec lat > exec lat! {wi.perfect_exec_lat} > {wi.exec_lat} ({wi.flops} ops)')
            logger.error(f'{wi}')

        self.log_exec_cycles(step, tid, wi.exec_lat, wi.perfect_exec_lat)
        self.flops += wi.flops

        if USE_C_TRACE:
            for tile in wi.read_trace:
                c_model.tile_read_trace(
                    self.l1[tid], self.dest_maps[step], tile, tid)

        else:
            for l in wi.read_trace:
                if self.l1[tid].lookup(l):
                    self.l1[tid].insert(l)
                    continue
                self.l1[tid].insert(l)
                self.log_read(step, tid, l)

        self.rss[tid].append(self.l1[tid].get_accesses() * self.arch.line_size)
        self.kwstats['l1_accesses'] += self.l1[tid].get_accesses()
        self.kwstats['l1_hits'] += self.l1[tid].get_hits()

        self.cur_step[tid] += 1

    def step(self):
        if self.cancel is not None and self.cancel.value: raise KeyboardInterrupt()

        traffic = self.noc_sim_func(self.arch, self.kwstats, self.global_step, self)
        net_latency = int(np.max(traffic) / self.arch.noc_ports_per_dir)
        self.total_traffic += traffic

        max_exec_cyc = int(max(self.exec_cycles[self.global_step]))
        perfect_exec_cyc = int(max(self.perfect_exec_cycles[self.global_step]))

        for i, (cs, ns) in enumerate(self.arch.scales):
            if cs == 0:
                max_exec_cyc_i = 0
                perfect_exec_cyc_i = 0
            else:
                max_exec_cyc_i = max_exec_cyc / cs
                perfect_exec_cyc_i = perfect_exec_cyc / cs

            if ns == 0: net_latency_i = 0
            else: net_latency_i = net_latency / ns

            if logger.isEnabledFor(logging.DEBUG) and cs == 1.0 and ns == 1.0:
                cores_used = np.count_nonzero(self.exec_cycles[self.global_step])
                logger.debug(f'Step {self.global_step + 1}/{self.total_steps}: Cores used: {cores_used}, Exec latency: {self.compute_cyc[i]} cyc (best possible: {perfect_exec_cyc_i}), Noc latency: {net_latency_i} cyc')

            # Update compute_cyc after this line because compute is one time-step
            # behind prefetch.
            self.cycles[i] += max(self.compute_cyc[i], net_latency_i)
            self.compute_cyc[i] = max_exec_cyc_i

        # tiles_rss = np.array([rss[self.global_step] for rss in self.rss if len(rss) > self.global_step])
        # avg_rss = np.mean(tiles_rss, axis=0)
        # max_rss = np.max(tiles_rss, axis=0)
        # self.kwstats['rss'].append((avg_rss, max_rss))

        # if logger.isEnabledFor(logging.DEBUG):
        #     logger.debug(f'+ Max RSS = {max_rss}, Avg RSS = {avg_rss}')

        if self.lock is not None and self.counter is not None:
            with self.lock: self.counter.value += 1

        del self.dest_maps[self.global_step]
        self.dest_maps[self.global_step] = None
        self.global_step += 1

    def drain(self):
        for i, (cs, ns) in enumerate(self.arch.scales):
            self.cycles[i] += self.compute_cyc[i]

    def barrier(self):
        max_step = max(self.cur_step)
        for tid in range(len(self.cur_step)): self.cur_step[tid] = max_step

    @property
    def nsteps(self): return max(self.cur_step)


def simulate_noc(arch : Arch, kwstats : dict, step : int, sim : SimBase):
    """Default NoC simulator."""
    raise NotImplementedError()

@functools.singledispatch
def num_steps(
    arch : Arch,
    op : Operator,
    **kwargs
):
    """Count the number of steps needed to execute an operator.

    N.B. This is mainly used to count total steps ahead of time for progress-
    reporting purposes. It serves no functional purpose.
    """
    sim = StepCounter(arch)
    place_op(arch, op, sim, check_flops=False)

    logger.debug(f'Counted {sim.nsteps} steps for op = {op}')
    return sim.nsteps

def common_sim(
    arch : Arch,
    op : Operator,
    ex_sim_cls = Sim,
    noc_sim_func : Callable = simulate_noc,
    counter : multiprocessing.Value = None,
    lock : multiprocessing.Lock = None,
    cancel : multiprocessing.Value = None,
    num_steps : int = None
):
    """Primary entry point for most simulator.

    N.B. This function will be called for most architectures. There could be a
    reason to write a custom sim function for a particular architecture in the
    future if there are some wacky features we want to model.
    """
    sim = ex_sim_cls(arch, noc_sim_func, num_steps, lock, counter, cancel)
    logger.debug(f'Simulating {num_steps} steps...')
    place_op(arch, op, sim)
    sim.drain()
    return SimResult(sim.nsteps, tuple(sim.cycles), sim.total_traffic, sim.kwstats)
