from dataclasses import dataclass
from typing import Callable
import numpy as np
import logging
import time

from ...common import *

from ..common import *
from .. import noc

from . import oracle

logger = logging.getLogger(__name__)


class HierSim(Sim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2 = c_model.CacheVector([
            c_model.Cache(
                self.arch.l2.nset(self.arch.line_size),
                self.arch.l2.assoc,
                self.arch.l2.lbits(self.arch.line_size))
            for _ in range(self.arch.ngroups)
        ])
        self.kwstats['l2_accesses'] = 0
        self.kwstats['l2_hits'] = 0

def simulate_hier_noc(arch : HierArch, kwstats : dict, step : int, sim : HierSim):
    dest_map = sim.dest_maps.get(step, None)
    if dest_map is None: return

    t0 = time.perf_counter()
    net = noc.Noc.from_arch(arch)
    l2_dl = c_model.DestList()

    l2_accesses = 0
    l2_hits = 0
    llc_accesses = 0

    # 1. For each line, determine if the local L2 needs it, and then count
    #    L2 -> L1 traffic.
    for l, mask in dest_map.dests.items():
        gids = set(arch.tid_to_gid(tid) for tid in get_dest_tids(arch, mask))
        groups = {gid: [] for gid in gids}

        for tid in get_dest_tids(arch, mask):
            gid = arch.tid_to_gid(tid)
            groups[gid].append(tid)

        for gid, tids in groups.items():
            if len(tids) == 0: continue
            l2_accesses += 1
            hit = sim.l2[gid].lookup(l)
            sim.l2[gid].insert(l)
            if not hit: l2_dl.set(l, arch.addr_l2_tid(l, gid))
            else: l2_hits += 1

            net.count_multiroute(
                arch.addr_l2_coords(l, gid),
                [arch.tile_coords(tid) for tid in tids],
                2 if arch.line_size == 64 else 1)

    # 2. Now that we've pruned the dest list down to the L2s that need the
    #    lines, we can count LLC -> L2 traffic.
    for l, mask in l2_dl.dests.items():
        llc_accesses += 1
        net.count_multiroute(
            arch.addr_llc_coords(l),
            get_dests(arch, mask),
            2 if arch.line_size == 64 else 1)

    t1 = time.perf_counter()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')

    traffic = noc.zero_traffic(arch)
    traffic[:, :, :] += net.to_numpy()

    sim.kwstats['l2_accesses'] += l2_accesses
    sim.kwstats['l2_hits'] += l2_hits
    sim.kwstats['llc_accesses'] += llc_accesses

    return traffic


def simulate_hier_noc2(arch : HierArch, kwstats : dict, step : int, sim : HierSim):
    dest_map = sim.dest_maps.get(step, None)
    if dest_map is None: return
    sim.kwstats['llc_accesses'] += len(dest_map)
    for i in range(arch.ngroups):
        sim.l2[i].reset_stats()

    t0 = time.perf_counter()
    traffic = noc.zero_traffic(arch)

    c_model.hier_traffic(
        arch.l1.lbits((arch.line_size)),
        arch.nrows, arch.ncols,
        arch.grows, arch.gcols,
        dest_map,
        traffic,
        sim.l2,
        2 if arch.line_size == 64 else 1)

    for i in range(arch.ngroups):
        sim.kwstats['l2_accesses'] += sim.l2[i].get_accesses()
        sim.kwstats['l2_hits'] += sim.l2[i].get_hits()

    t1 = time.perf_counter()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')

    return traffic

@register_sim(HierArch)
def hier_sim(arch : HierArch, op : Operator, *args, **kwargs):
    return common_sim(
        arch, op, *args, ex_sim_cls=HierSim, noc_sim_func=simulate_hier_noc2, **kwargs)
