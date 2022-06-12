from dataclasses import dataclass
import numpy as np
import logging
import time

from ..common import *

from .common import *
from . import cache
from . import noc


logger = logging.getLogger(__name__)


@register_sim(OracleArch)
def oracle_sim(arch : Arch, op : Operator, placement_mode='naive', l1_capacity=16384, l1_assoc=4, randomize_llc=False):
    cycles = 0

    if randomize_llc:
        llc_addr_map = list(range(arch.ntiles))
        random.shuffle(llc_addr_map)
    else:
        llc_addr_map = None

    def tile_coords(tid):
        assert tid >= 0 and tid < arch.ntiles
        return (tid // arch.ncols), (tid % arch.ncols)

    def addr_llc_coords(addr : int):
        line = addr >> 6
        tid = line & (arch.ntiles - 1)
        if llc_addr_map is not None:
            return tile_coords(llc_addr_map[tid])
        else:
            return tile_coords(tid)


    wl = place_op(placement_mode, arch, op)
    l1_nway = int(l1_assoc)
    l1_nset = int(l1_capacity / 64 / l1_nway)
    l1 = [
        [cache.Cache(l1_nset, l1_nway, 6) for _ in range(arch.ncols)]
        for _ in range(arch.nrows)
    ]

    compute_cyc = 0

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'Simulating {wl.nsteps} steps with {wl.flops} flops...')

    traffic = noc.zero_traffic(arch, wl.nsteps)

    for step in range(wl.nsteps):
        logger.debug(f'Step {step}')
        oracle_map = dict()

        exec_cyc = []
        idle_tiles = 0
        flops = 0

        accesses = 0
        hits = 0

        # for tid, tile in enumerate(wl.flattiles):
        for r in range(arch.nrows):
            for c in range(arch.ncols):
                tile = wl[r, c]
                l1[r][c].reset()
                if step >= len(tile):
                    idle_tiles += 1
                    continue
                exec_cyc.append(tile[step].exec_lat)
                flops += tile[step].flops
                for l in tile[step].read_trace:
                    if l1[r][c].lookup(l):
                        l1[r][c].insert(l)
                        continue
                    l1[r][c].insert(l)
                    if l not in oracle_map: oracle_map[l] = []
                    oracle_map[l].append((r, c))

                accesses += l1[r][c].get_accesses()
                hits += l1[r][c].get_hits()

        max_exec_cyc = max(exec_cyc)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'+ Max exec cycles: {max_exec_cyc}')
            logger.debug(f'+ Avg exec cycles: {np.average(exec_cyc)}')
            logger.debug(f'+ Idle tiles: {idle_tiles}')
            logger.debug(f'+ Step flops: {flops}')
            logger.debug(f'+ L1 Hit Rate: {hits} / {accesses} = {np.round(hits / max(accesses, 1), 2) * 100}%')

            dsts = [len(d) for _, d in oracle_map.items()]
            if len(dsts) > 0:
                avg_dests_per_line = np.average(dsts)
                logger.debug(f'+ Avg dests per line: {avg_dests_per_line}')

            logger.debug(f'+ # lines transmitted: {len(oracle_map)}')

        t0 = time.perf_counter()
        net = noc.Noc.from_arch(arch)
        for line, dests in oracle_map.items():
            r, c = addr_llc_coords(line)
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

        traffic[step, :, :, :] = net.to_numpy()

        t1 = time.perf_counter()

        net_latency = np.max(traffic[step, :, :, :]) / arch.noc_ports_per_dir

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'+ Noc simulation took {t1 - t0}s')
            logger.debug(f'+ Exec latency: {compute_cyc} cyc')
            logger.debug(f'+ Noc latency: {net_latency} cyc')

        cycles += max(compute_cyc, net_latency)
        compute_cyc = max_exec_cyc


    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Compute Drain')
        logger.debug(f'+ Exec latency: {compute_cyc}')

    cycles += compute_cyc

    return SimResult(wl.nsteps, cycles, traffic)

