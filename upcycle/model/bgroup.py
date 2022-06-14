from dataclasses import dataclass
import functools
import numpy as np
import logging
import time

from ..common import *

from .common import *
from . import cache
from . import noc


logger = logging.getLogger(__name__)

@functools.lru_cache
def make_rally_points(arch : BgroupArch):

    ngr = arch.nrows // arch.grows
    ngc = arch.ncols // arch.gcols

    c_off_r = 0 #arch.grows // 2
    c_off_c = 0 #arch.gcols // 2
    rally_points = dict()
    group_hops = dict()

    for gr in range(ngr):
        for gc in range(ngc):
            r, c = gr * arch.grows, gc * arch.gcols
            rpr, rpc = r + c_off_r, c + c_off_c
            rally_points[(gr, gc)] = (rpr, rpc)
            hops = []

            for orow in range(arch.grows):
                for ocol in range(arch.gcols):
                    rr, rc = r + orow, c + ocol
                    if rr == rpr and rc == rpc: continue
                    hops += noc.Noc.trace_route((rpr, rpc), (rr, rc))


            group_hops[(gr, gc)] = set(hops)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'+ Rally point: {(rpr, rpc)}, # Group Hops: {len(set(hops))}')

    return rally_points, group_hops

@functools.lru_cache
def trace_group_hops(arch : BgroupArch, group, off):
    gr, gc = group
    off_r, off_c = off
    rpr, rpc = gr * arch.grows + off_r, gc * arch.gcols + off_c
    hops = []

    for lr in range(arch.grows):
        for lc in range(arch.gcols):
            rr, rc = gr * arch.grows + lr, gc * arch.gcols + lc
            if rr == rpr and rc == rpc: continue
            hops += noc.Noc.trace_route((rpr, rpc), (rr, rc))

    return set(hops)

@dataclass
class BgroupNoc(noc.Noc):
    @staticmethod
    def from_arch(arch : BgroupArch):
        return BgroupNoc(arch, [
            [noc.Router(r, c) for c in range(arch.ncols)]
            for r in range(arch.nrows)
        ], arch.noc_ports_per_dir)

    def get_rally_point(self, group, off):
        gr, gc = group
        off_r, off_c = off
        return gr * self.arch.grows + off_r, gc * self.arch.gcols + off_c


def simulate_bgroup_noc(arch : Arch, kwstats : dict, dest_map : dict, addr_llc_coords : Callable):
    t0 = time.perf_counter()
    net = BgroupNoc.from_arch(arch)
    sum_dests = 0
    sum_groups = 0
    for line, dests in dest_map.items():
        r, c = addr_llc_coords(line)
        off = (np.random.randint(0, arch.grows), np.random.randint(0, arch.gcols))
        net[r, c].inject += 1

        groups = set()
        seen_hops = set[(noc.Router, noc.Router)]()

        for (dr, dc) in dests:
            # logger.debug(f'++ {(r, c)} -> {(dr, dc)}')
            net[dr, dc].eject += 1
            dgr = dr // arch.grows
            dgc = dc // arch.gcols
            groups.add((dgr, dgc))

        for (dgr, dgc) in groups:
            rr, rc = net.get_rally_point((dgr, dgc), off)
            # logger.debug(f'++ rally point: {(rr, rc)}')

            route = net.get_route((r, c), (rr, rc))

            for (rs, rd) in net.route_hops(route):
                if (rs, rd) in seen_hops: continue
                seen_hops.add((rs, rd))
                net.count_hop(rs, rd)
                net.count_hop(rs, rd)

            for (rs_coords, rd_coords) in trace_group_hops(arch, (dgr, dgc), off):
                rs = net.__getitem__(rs_coords)
                rd = net.__getitem__(rd_coords)
                # logger.debug(f'+++ group hop: {(rs, rd)}')
                net.count_hop(rs, rd)
                net.count_hop(rs, rd)

        sum_dests += len(dests)
        sum_groups += len(groups)

    t1 = time.perf_counter()
    avg_dests = np.round(sum_dests / max(1, len(dest_map)), 2)
    avg_groups = np.round(sum_groups / max(1, len(dest_map)), 2)

    if 'avg_dests' not in kwstats: kwstats['avg_dests'] = []
    kwstats['avg_dests'].append(avg_dests)

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