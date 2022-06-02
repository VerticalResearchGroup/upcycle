from dataclasses import dataclass

from ..common import *
from .. import ops

from .common import *
from . import cache
from . import noc

@register_soc(FlatMeshArch)
class FlatMeshSoc(Soc):
    def __init__(self, arch : Arch, l1_capacity=16384, l1_assoc=4, randomize_llc=False):
        super().__init__(arch)
        self.l1_capacity = l1_capacity
        self.l1_assoc = l1_assoc
        self._cycles = 0
        if randomize_llc:
            self.llc_addr_map = list(range(self.arch.ntiles))
            random.shuffle(self.llc_addr_map)
        else:
            self.llc_addr_map = None

    def tile_coords(self, tid):
        assert tid >= 0 and tid < self.arch.ntiles
        return (tid // self.arch.ncol), (tid % self.arch.ncol)

    def addr_llc_coords(self, addr : int):
        line = addr >> 6
        tid = line & (self.arch.ntiles - 1)
        if self.llc_addr_map is not None:
            return self.tile_coords(self.llc_addr_map[tid])
        else:
            return self.tile_coords(tid)

    def simulate(self, op : ops.Operator):
        gwl = place_op(self.placement_mode, self.arch, op)
        l1_nway = int(self.l1_assoc)
        l1_nset = int(self.l1_capacity / 64 / l1_nway)
        l1 = [cache.Cache(l1_nset, l1_nway, 6) for _ in range(self.arch.ntiles)]

        for step in range(gwl.nsteps):
            net = noc.Noc.from_arch(self.arch)
            for tid, tile in enumerate(gwl.tiles):
                sr, sc = self.tile_coords(tid)
                if step >= len(tile): continue
                wi = tile[step]
                for tt in wi.read_trace:
                    for l in tt.lines:
                        if l1[tid].lookup(l): continue
                        l1[tid].insert(l)
                        dr, dc = self.addr_llc_coords(l)
                        route = list(net.get_route((sr, sc), (dr, dc)))
                        net.count_route(route)
                        net.count_route(list(reversed(route)))
                        net.count_route(list(reversed(route)))
                        net.count_route(list(reversed(route)))
                        net.count_route(list(reversed(route)))

            self._cycles += net.latency

    @property
    def cycles(self): return self._cycles




