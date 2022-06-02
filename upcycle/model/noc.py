from dataclasses import dataclass

from ..common import *
from .common import *

@dataclass
class Router:
    r : int
    c : int

    inject : int = 0
    eject : int = 0

    in_north : int = 0
    in_south : int = 0
    in_east : int = 0
    in_west : int = 0

    out_north : int = 0
    out_south : int = 0
    out_east : int = 0
    out_west : int = 0


    @property
    def num_in(self): return self.in_north + self.in_south + self.in_east + self.in_west

    @property
    def num_out(self): return self.out_north + self.out_south + self.out_east + self.out_west


@dataclass
class Noc:
    arch : Arch
    routers : list[list[Router]]

    def __getitem__(self, coords : tuple[int, int]):
        return self.routers[coords[0]][coords[1]]

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

    def count_hop(self, rs : Router, rd : Router):
        if rs.r > rd.r:
            rs.out_south += 1
            rd.in_north += 1
        elif rs.r < rd.r:
            rs.out_north += 1
            rd.in_south += 1
        elif rs.c > rd.c:
            rs.out_west += 1
            rd.in_east += 1
        else:
            rs.out_east += 1
            rd.in_west += 1

    def count_route(self, route):
        route[0].inject += 1
        route[-1].eject += 1
        for i in range(len(route) - 1):
            self.count_hop(route[i], route[i + 1])

    @property
    def latency(self):
        max_out = max([
            self[r, c].num_out
            for r in range(self.arch.nrow)
            for c in range(self.arch.ncol)
        ])

        return max(max_out // 4, 1)

    @property
    def enb(self):
        num_reqs = sum([
            self[r, c].inject
            for r in range(self.arch.nrow)
            for c in range(self.arch.ncol)
        ])

        read_bytes = num_reqs * 64
        return read_bytes / self.latency


    @staticmethod
    def from_arch(arch : Arch):
        return Noc(arch, [
            [Router(r, c) for c in range(arch.ncol)]
            for r in range(arch.nrow)
        ])

