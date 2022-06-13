from dataclasses import dataclass
import numpy as np
import logging
import time

from ..common import *

from .common import *
from . import cache
from . import noc


logger = logging.getLogger(__name__)

def simulate_bgroup_noc(arch : Arch, kwstats : dict, dest_map : dict, addr_llc_coords : Callable):
    t0 = time.perf_counter()
    net = noc.Noc.from_arch(arch)
    sum_groups = 0
    for line, dests in dest_map.items():
        r, c = addr_llc_coords(line)

        groups = set()
        for (dr, dc) in dests:
            dgr = dr // arch.grows
            dgc = dc // arch.gcols
            gid = dgr * (arch.ncols / arch.gcols) + dgc
            groups.add(gid)

        sum_groups += len(groups)

        net[r, c].inject += 1

        for (dr, dc) in dests: net[dr, dc].eject += 1

        routes = [
            net.get_route((r, c), (dr, dc))
            for (dr, dc) in dests
        ]

        seen_hops = set[(noc.Router, noc.Router)]()
        for route in routes:
            for (rs, rd) in net.route_hops(route):
                if (rs, rd) in seen_hops: continue
                seen_hops.add((rs, rd))
                net.count_hop(rs, rd)
                net.count_hop(rs, rd)


    t1 = time.perf_counter()
    avg_groups = sum_groups / max(1, len(dest_map))

    if 'avg_groups' not in kwstats: kwstats['avg_groups'] = []
    kwstats['avg_groups'].append(avg_groups)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')
        logger.debug(f'+ Average groups per line: {avg_groups}')

    return net.to_numpy()

@register_sim(BgroupArch)
def bgroup_sim(arch : Arch, op : Operator, *args, **kwargs):
    return common_sim(
        arch, op, *args, noc_sim_func=simulate_bgroup_noc, **kwargs)
