from dataclasses import dataclass
from typing import Callable
import numpy as np
import logging
import time

from ...common import *

from ..common import *
from .. import noc


logger = logging.getLogger(__name__)

def simulate_oracle_noc(arch : Arch, kwstats : dict, step : int, sim : SimBase):
    dest_map = sim.dest_maps.get(step, None)
    t0 = time.perf_counter()
    net = noc.Noc.from_arch(arch)

    if dest_map is not None:
        for line, mask in dest_map.dests.items():
            net.count_multiroute(
                arch.addr_llc_coords(line),
                get_dests(arch, mask),
                2 if arch.line_size == 64 else 1)
    t1 = time.perf_counter()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')

    return net.to_numpy()

def simulate_oracle_noc2(arch : Arch, kwstats : dict, dl : c_model.DestList):
    t0 = time.perf_counter()
    traffic = np.zeros((arch.nrows, arch.ncols, noc.NocDir.DIRMAX), dtype=np.uint64)

    if dl is not None:
        c_model.oracle_traffic(arch.l1.lbits((arch.line_size)), arch.nrows, arch.ncols, dl, traffic)

    t1 = time.perf_counter()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'+ Noc simulation took {t1 - t0}s')

    return traffic


@register_sim(OracleArch)
def oracle_sim(arch : Arch, op : Operator, *args, **kwargs):
    return common_sim(
        arch, op, *args, noc_sim_func=simulate_oracle_noc, **kwargs)
