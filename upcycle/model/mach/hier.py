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

@noc.zero_traffic.register(HierArch)
def _(arch : HierArch):
    return np.zeros(
        (arch.nrows, arch.ncols, noc.NocDir.DIRMAX + 1), dtype=np.uint32)

def simulate_hier_oracle_noc(arch : HierArch, kwstats : dict, step : int, sim : HierSim):
    dest_map = sim.dest_maps.get(step, None)
    if dest_map is None: return

    t0 = time.perf_counter()
    net = noc.Noc.from_arch(arch)
    l2_dl = c_model.DestList()
    l2_broadcasts = [0 for _ in range(arch.ngroups)]

    for l, mask in dest_map.dests.items():
        groups = set(arch.tid_to_gid(tid) for tid in get_dest_tids(arch, mask))

        for gid in groups:
            if sim.l2[gid].lookup(l):
                sim.l2[gid].insert(l)
                continue
            sim.l2[gid].insert(l)
            l2_dl.set(l, gid)

    for line, mask in l2_dl.dests.items():
        r, c = arch.addr_llc_coords(line)
        net[r, c].inject += 1

        for (dr, dc) in get_dests(arch, mask): net[dr, dc].eject += 1

        routes = [
            net.get_route((r, c), (dr, dc))
            for (dr, dc) in get_dests(arch, mask)
        ]

        seen_hops = set[(noc.Router, noc.Router)]()
        for route in routes:
            for (rs, rd) in net.route_hops(route):
                if (rs, rd) in seen_hops: continue
                seen_hops.add((rs, rd))
                net.count_hop(rs, rd)
                if arch.line_size == 64: net.count_hop(rs, rd)

    t1 = time.perf_counter()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')

    traffic = noc.zero_traffic(arch)
    traffic[:, :, 0:noc.NocDir.DIRMAX] += net.to_numpy()

    for gid in range(arch.ngroups):
        r, c = arch.group_coords(gid)
        traffic[r, c, noc.NocDir.DIRMAX] = 0 #l2_broadcasts[gid]

    return traffic

@register_sim(HierArch)
def hier_sim(arch : HierArch, op : Operator, *args, **kwargs):
    return common_sim(
        arch, op, *args, ex_sim_cls=HierSim, noc_sim_func=simulate_hier_oracle_noc, **kwargs)
