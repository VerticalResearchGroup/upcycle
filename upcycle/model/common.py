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
        # N.B. We use Python's __getitem__ as a frontend for all this indexing
        # nonsense for cosmetic reasons. This way we can make a tensor and then
        # write something like A[1:2, 5:, :] and it will produce all the
        # addresses for that slice.
        global USE_C_TRACE
        if USE_C_TRACE: return self.c_trace(_idx)
        else: return self.py_trace(_idx)

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

    @property
    def exec_lat(self): return self.perfect_exec_lat

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
    def __init__(self, arch):
        self.arch = arch

    def place_work(self, tid, wl : list[WorkItem]): raise NotImplementedError()

    def flatmap_place(self, vtiles : list[list[WorkItem]], offset=0, bbox=None, randomize=False):
        if randomize: shuffle(vtiles)

        logger.debug(f'flatmap_place: {len(vtiles)} vtiles')

        if bbox is not None: raise DeprecationWarning('bbox is deprecated')

        i = 0
        for tid, n in enumerate(blkdiv(len(vtiles), self.arch.ntiles)):
            for vtile in vtiles[i : i + n]:
                self.place_work(tid, vtile)

            i += n
        return len(vtiles)

    def map2d_place(self, vtiles : list[list[list[WorkItem]]]):
        trows, tcols = len(vtiles), len(vtiles[0])

        vrow = 0
        for r, m in enumerate(blkdiv(trows, self.arch.nrows)):
            for tilerow in vtiles[vrow : vrow + m]:
                vcol = 0
                for c, n in enumerate(blkdiv(tcols, self.arch.ncols)):
                    for tile in tilerow[vcol : vcol + n]:
                        tid = self.arch.coords_tile(r, c)
                        self.place_work(tid, tile)
                    vcol += n

            vrow += m

        return trows * tcols

def place_op(mode : str, arch : Arch, op : Operator, sim : SimBase, check_flops=True):
    global placement_funcs
    logger.debug(f'place_op(mode={mode}): {arch} {op}')
    placement_funcs[(mode, type(arch), type(op))](arch, op, sim)
    if check_flops and op.flops != sim.flops:
        logger.error(f'Placement produced different number of FLOPs! (op={op.flops} != wl={sim.flops}, wl/op={sim.flops / op.flops}x)')

class StepCounter(SimBase):
    """Dummy sim object used to count number of steps for a layer.

    N.B. This is used to predict runtime ahead of execution. See num_steps()
    below.
    """
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
        self.rss = [[] for _ in range(arch.ntiles)]
        self.l1_accesses = 0
        self.l1_hits = 0
        self.flops = 0
        self.cur_step = [0 for _ in range(arch.ntiles)]
        self.l1 = [
            c_model.Cache(
                arch.l1.nset(arch.line_size),
                arch.l1.assoc,
                arch.l1.lbits(arch.line_size))
            for _ in range(arch.ntiles)
        ]

    def log_exec_cycles(self, step, tid, ncycles):
        if step not in self.exec_cycles:
            self.exec_cycles[step] = np.zeros(self.arch.ntiles, dtype=np.int32)
        self.exec_cycles[step][tid] += ncycles

    def log_read(self, step, tid, laddr):
        if step not in self.dest_maps:
            self.dest_maps[step] = c_model.DestList()
        self.dest_maps[step].set(laddr, tid)

    def place_work(self, tid, wl : list[WorkItem]):
        global USE_C_TRACE
        tile_cur_step = self.cur_step[tid]
        self.cur_step[tid] += len(wl)

        for i, wi in enumerate(wl):
            self.l1[tid].reset()
            step = tile_cur_step + i

            if step not in self.dest_maps:
                self.dest_maps[step] = c_model.DestList()

            # if logger.isEnabledFor(logging.DEBUG):
            if wi.perfect_exec_lat > wi.exec_lat:
                logger.error(f'Perfect exec lat > exec lat! {wi.perfect_exec_lat} > {wi.exec_lat} ({wi.flops} ops)')
                logger.error(f'{wi}')

            exec_lat = wi.perfect_exec_lat if self.arch.perfect_compute else wi.exec_lat

            self.log_exec_cycles(step, tid, exec_lat)
            self.flops += wi.flops

            if USE_C_TRACE:
                for tile in wi.read_trace:
                    c_model.tile_read_trace(
                        self.l1[tid],
                        self.dest_maps[step],
                        tile,
                        tid)

            else:
                for l in wi.read_trace:
                    if self.l1[tid].lookup(l):
                        self.l1[tid].insert(l)
                        continue
                    self.l1[tid].insert(l)
                    self.log_read(step, tid, l)

            self.rss[tid].append(self.l1[tid].get_accesses() * self.arch.line_size)
            self.l1_accesses += self.l1[tid].get_accesses()
            self.l1_hits += self.l1[tid].get_hits()

    @property
    def nsteps(self): return max(self.cur_step)


def simulate_noc(arch : Arch, kwstats : dict, step : int, sim : SimBase):
    """Default NoC simulator."""
    raise NotImplementedError()

@functools.singledispatch
def num_steps(
    arch : Arch,
    op : Operator,
    placement_mode : str = 'naive',
    **kwargs
):
    """Count the number of steps needed to execute an operator.

    N.B. This is mainly used to count total steps ahead of time for progress-
    reporting purposes. It serves no functional purpose.
    """
    sim = StepCounter(arch)
    place_op(placement_mode, arch, op, sim, check_flops=False)
    return sim.nsteps

def common_sim(
    arch : Arch,
    op : Operator,
    ex_sim_cls = Sim,
    noc_sim_func : Callable = simulate_noc,
    placement_mode : str = 'naive',
    counter : multiprocessing.Value = None,
    lock : multiprocessing.Lock = None,
):
    """Primary entry point for most simulator.

    N.B. This function will be called for most architectures. There could be a
    reason to write a custom sim function for a particular architecture in the
    future if there are some wacky features we want to model.
    """
    sim = ex_sim_cls(arch)
    kwstats = {}

    # place_op will call the placement function for the given operator an
    # collect step-wise information such as the read trace.
    place_op(placement_mode, arch, op, sim)

    logger.debug(f'Simulating {sim.nsteps} steps with {sim.flops} flops...')
    total_traffic = noc.zero_traffic(arch)
    cycles = 0
    compute_cyc = 0
    nsteps = sim.nsteps

    kwstats['rss'] = []

    # Main simulator loop. By now we've already done all the core modeling, so
    # we just need to simulate the traffic for each step.
    for step in range(nsteps):
        max_exec_cyc = max(sim.exec_cycles[step])
        traffic = noc_sim_func(arch, kwstats, step, sim)
        net_latency = np.max(traffic) / arch.noc_ports_per_dir
        total_traffic += traffic

        if logger.isEnabledFor(logging.DEBUG):
            cores_used = np.count_nonzero(sim.exec_cycles[step])
            logger.debug(f'Step {step + 1}/{sim.nsteps}: Cores used: {cores_used}, Exec latency: {compute_cyc} cyc, Noc latency: {net_latency} cyc')

        # Update compute_cyc after this line because compute is one time-step
        # behind prefetch.
        cycles += max(compute_cyc, net_latency)
        compute_cyc = max_exec_cyc

        tiles_rss = np.array([rss[step] for rss in sim.rss if len(rss) > step])
        avg_rss = np.mean(tiles_rss, axis=0)
        max_rss = np.max(tiles_rss, axis=0)
        kwstats['rss'].append((avg_rss, max_rss))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'+ Max RSS = {max_rss}, Avg RSS = {avg_rss}')

        if lock is not None and counter is not None:
            with lock: counter.value += 1

    logger.debug(f'Compute drain latency: {compute_cyc}')

    # I'm not sure if this actually helps at all, but here's some stuff that
    # might be useful for reducing memory usage.
    if aggressive_mem():
        del kwstats
        del sim
        del total_traffic
        total_traffic = None
        kwstats = {}

    cycles += compute_cyc
    return SimResult(nsteps, cycles, total_traffic, kwstats)
