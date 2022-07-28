from dataclasses import dataclass
from typing import Callable
import numpy as np
import logging
import time

from ...common import *

from ..common import *
from .. import noc


logger = logging.getLogger(__name__)

def simulate_oracle_noc(arch : Arch, kwstats : dict, dest_map : dict):
    t0 = time.perf_counter()
    net = noc.Noc.from_arch(arch)

    if dest_map is not None:
        for line, mask in dest_map.dests.items():
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

    return net.to_numpy()

def simulate_oracle_noc2(arch : Arch, kwstats : dict, dl : c_model.DestList):
    t0 = time.perf_counter()
    traffic = np.zeros((arch.nrows, arch.ncols, noc.NocDir.DIRMAX), dtype=np.uint64)

    if dl is not None:
        c_model.oracle_traffic(arch.lbits, arch.nrows, arch.ncols, dl, traffic)

    t1 = time.perf_counter()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')

    return traffic


@register_sim(OracleArch)
def oracle_sim(arch : Arch, op : Operator, *args, **kwargs):
    return common_sim(
        arch, op, *args, noc_sim_func=simulate_oracle_noc, **kwargs)
