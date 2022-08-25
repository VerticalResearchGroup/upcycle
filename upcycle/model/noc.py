from dataclasses import dataclass
import functools
from enum import IntEnum
import logging

from ..common import *
from .common import *

logger = logging.getLogger(__name__)

class NocDir(IntEnum):
    OUT_NORTH = 0
    OUT_SOUTH = 1
    OUT_EAST = 2
    OUT_WEST = 3
    INJECT = 4
    EJECT = 5
    DIRMAX = 6

@functools.singledispatch
def zero_traffic(arch : Arch):
    return np.zeros((arch.nrows, arch.ncols, NocDir.DIRMAX), dtype=np.uint32)

@dataclass
class Router:
    r : int
    c : int

    inject : int = 0
    eject : int = 0

    out_north : int = 0
    out_south : int = 0
    out_east : int = 0
    out_west : int = 0

    def __hash__(self): return hash((self.r, self.c))


@dataclass
class Noc:
    arch : Arch
    routers : list[list[Router]]
    ports_per_dir : int

    def __getitem__(self, coords : tuple[int, int]):
        return self.routers[coords[0]][coords[1]]

    @staticmethod
    def trace_route(src : tuple[int, int], dst : tuple[int, int]):
        r, c = src
        route = []
        prev = (r, c)

        while c != dst[1]:
            c += 1 if c < dst[1] else -1
            cur = (r, c)
            route.append((prev, cur))
            prev = cur

        while r != dst[0]:
            r += 1 if r < dst[0] else -1
            cur = (r, c)
            route.append((prev, cur))
            prev = cur

        return route

    def get_route(self, src : tuple[int, int], dst : tuple[int, int]):
        r, c = src
        route = [self[r, c]]

        while c != dst[1]:
            c += 1 if c < dst[1] else -1
            route.append(self[r, c])

        while r != dst[0]:
            r += 1 if r < dst[0] else -1
            route.append(self[r, c])

        return route

    def count_hop(self, rs : Router, rd : Router, n : int = 1):
        if rs.r > rd.r: rs.out_south += n
        elif rs.r < rd.r: rs.out_north += n
        elif rs.c > rd.c: rs.out_west += n
        else: rs.out_east += n

    def count_route(self, route):
        route[0].inject += 1
        route[-1].eject += 1
        for i in range(len(route) - 1):
            self.count_hop(route[i], route[i + 1])

    def count_multiroute(self, src : tuple[int, int], dests : list[tuple[int, int]], n : int = 1):
        r, c = src
        self[r, c].inject += 1
        for (dr, dc) in dests: self[dr, dc].eject += 1
        routes = [self.get_route((r, c), (dr, dc)) for (dr, dc) in dests]

        seen_hops = set[(Router, Router)]()
        for route in routes:
            for (rs, rd) in self.route_hops(route):
                if (rs, rd) in seen_hops: continue
                seen_hops.add((rs, rd))
                self.count_hop(rs, rd, n)

    def route_hops(self, route):
        for i in range(len(route) - 1):
            yield (route[i], route[i + 1])

    @property
    def latency(self):
        max_out = max([
            self[r, c].num_out
            for r in range(self.arch.nrows)
            for c in range(self.arch.ncols)
        ])

        return max(max_out // self.ports_per_dir // 5, 1)

    @property
    def enb(self):
        num_reqs = sum([
            self[r, c].inject
            for r in range(self.arch.nrows)
            for c in range(self.arch.ncols)
        ])

        read_bytes = num_reqs * 64
        return read_bytes / self.latency

    @property
    def total_hops(self):
        return sum([
            self[r, c].num_out
            for r in range(self.arch.nrows)
            for c in range(self.arch.ncols)
        ])

    @staticmethod
    def from_arch(arch : Arch):
        return Noc(arch, [
            [Router(r, c) for c in range(arch.ncols)]
            for r in range(arch.nrows)
        ], arch.noc_ports_per_dir)

    def to_numpy(self):
        return np.array([
            [
                [self[r, c].out_north, self[r, c].out_south, self[r, c].out_east, self[r, c].out_west, self[r, c].inject, self[r, c].eject]
                for c in range(self.arch.ncols)
            ]
            for r in range(self.arch.nrows)
        ], dtype=np.uint32)

