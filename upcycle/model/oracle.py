from dataclasses import dataclass
import numpy as np
import logging

from ..common import *

from .common import *
from . import cache
from . import noc

@register_soc(OracleArch)
class OracleSoc(Soc):
    def __init__(self, arch : Arch, l1_capacity=16384, l1_assoc=4, randomize_llc=False):
        super().__init__(arch)
        self.nocs = []
        self.l1_capacity = l1_capacity
        self.l1_assoc = l1_assoc
        self._nsteps = 0
        self._cycles = 0
        if randomize_llc:
            self.llc_addr_map = list(range(self.arch.ntiles))
            random.shuffle(self.llc_addr_map)
        else:
            self.llc_addr_map = None

    def tile_coords(self, tid):
        assert tid >= 0 and tid < self.arch.ntiles
        return (tid // self.arch.ncols), (tid % self.arch.ncols)

    def addr_llc_coords(self, addr : int):
        line = addr >> 6
        tid = line & (self.arch.ntiles - 1)
        if self.llc_addr_map is not None:
            return self.tile_coords(self.llc_addr_map[tid])
        else:
            return self.tile_coords(tid)

    def simulate(self, op : Operator):
        wl = place_op(self.placement_mode, self.arch, op)
        l1_nway = int(self.l1_assoc)
        l1_nset = int(self.l1_capacity / 64 / l1_nway)
        l1 = [
            [cache.Cache(l1_nset, l1_nway, 6) for _ in range(self.arch.ncols)]
            for _ in range(self.arch.nrows)
        ]

        compute_cyc = 0
        logging.debug(f'Simulating {wl.nsteps} steps with {wl.flops} flops...')

        for step in range(wl.nsteps):
            logging.debug(f'Step {step}')
            net = noc.Noc.from_arch(self.arch)
            self.nocs.append(net)
            oracle_map = dict()

            max_exec_cyc = 0

            # for tid, tile in enumerate(wl.flattiles):
            for r in range(self.arch.nrows):
                for c in range(self.arch.ncols):
                    tile = wl[r, c]
                    if step >= len(tile): continue
                    max_exec_cyc = max(tile[step].exec_lat, max_exec_cyc)
                    for l in tile[step].read_trace:
                        if l1[r][c].lookup(l): continue
                        l1[r][c].insert(l)
                        if l not in oracle_map: oracle_map[l] = []
                        oracle_map[l].append((r, c))

            dsts = [len(d) for _, d in oracle_map.items()]
            if len(dsts) > 0:
                avg_dests_per_line = np.average(dsts)
                logging.debug(f'+ Avg dests per line: {avg_dests_per_line}')

            for line, dests in oracle_map.items():
                r, c = self.addr_llc_coords(line)
                net[r, c].inject += 1

                for (dr, dc) in dests:
                    net[dr, dc].eject += 1

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

            logging.debug(f'+ Exec latency: {compute_cyc}')

            self._nsteps += 1
            self._cycles += max(compute_cyc, net.latency)
            compute_cyc = max_exec_cyc

        logging.debug('Compute Drain')
        logging.debug(f'+ Exec latency: {compute_cyc}')
        self._cycles += compute_cyc


    def noc(self, step : int): return self.nocs[step]

    @property
    def nsteps(self): return self._nsteps

    @property
    def cycles(self): return self._cycles



