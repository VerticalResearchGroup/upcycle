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
    def __init__(self, arch : HierArch):
        super().__init__(arch)
        self.l2 = [
            c_model.Cache(
                arch.l2.nset(arch.line_size),
                arch.l2.assoc,
                arch.l2.lbits(arch.line_size))
            for _ in range(arch.ngroups)
        ]

def simulate_hier_oracle_noc(arch : HierArch, kwstats : dict, step : int, sim : HierSim):
    dest_map = sim.dest_maps.get(step, None)
    if dest_map is None: return

    t0 = time.perf_counter()
    net = noc.Noc.from_arch(arch)
    l2_dl = c_model.DestList()

    # 1. For each line, determine if the local L2 needs it, and then count
    #    L2 -> L1 traffic.
    for l, mask in dest_map.dests.items():
        gids = set(arch.tid_to_gid(tid) for tid in get_dest_tids(arch, mask))
        groups = {gid: [] for gid in gids}

        for tid in get_dest_tids(arch, mask):
            gid = arch.tid_to_gid(tid)
            groups[gid].append(tid)

        for gid, tids in groups.items():
            hit = sim.l2[gid].lookup(l)
            sim.l2[gid].insert(l)
            if not hit: l2_dl.set(l, gid)

            net.count_multiroute(
                arch.addr_l2_coords(l, gid),
                [arch.tile_coords(tid) for tid in tids],
                2 if arch.line_size == 64 else 1)

    # 2. Now that we've pruned the dest list down to the L2s that need the
    #    lines, we can count LLC -> L2 traffic.
    for l, mask in l2_dl.dests.items():
        net.count_multiroute(
            arch.addr_llc_coords(l),
            get_dests(arch, mask),
            2 if arch.line_size == 64 else 1)

    t1 = time.perf_counter()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')

    traffic = noc.zero_traffic(arch)
    traffic[:, :, :] += net.to_numpy()


    return traffic

@register_sim(HierArch)
def hier_sim(arch : HierArch, op : Operator, *args, **kwargs):
    return common_sim(
        arch, op, *args, ex_sim_cls=HierSim, noc_sim_func=simulate_hier_oracle_noc, **kwargs)
