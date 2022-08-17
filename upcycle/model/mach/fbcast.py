from dataclasses import dataclass
from typing import Callable
import numpy as np
import logging
import time

from ...common import *

from ..common import *
from .. import noc


logger = logging.getLogger(__name__)

def simulate_fbcast_noc(arch : FbcastArch, kwstats : dict, step : int, sim : SimBase):
    dest_map = sim.dest_maps.get(step, None)
    t0 = time.perf_counter()
    net = noc.Noc.from_arch(arch)
    max_dests = arch.max_dests

    if dest_map is not None:
        for line, mask in dest_map.dests.items():
            r, c = arch.addr_llc_coords(line)
            all_dests = sorted(get_dest_tids(arch, mask))

            # partition destinations into groups with max size max_dests
            dest_groups = [
                all_dests[i : i + max_dests]
                for i in range(0, len(all_dests), max_dests)
            ]

            for dests in dest_groups:
                dests = list(map(lambda x: arch.tile_coords(x), dests))
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
                        if arch.line_size == 64: net.count_hop(rs, rd)

    t1 = time.perf_counter()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')

    return net.to_numpy()



@register_sim(FbcastArch)
def oracle_sim(arch : Arch, op : Operator, *args, **kwargs):
    return common_sim(
        arch, op, *args, noc_sim_func=simulate_fbcast_noc, **kwargs)
