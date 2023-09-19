from dataclasses import dataclass
from . import coeffs

@dataclass
class Component:
    node : int
    area : float
    delay : float
    power : float

    @property
    def freq(self): return 1 / self.delay

    def scale_isopower(self, to_node : int):
        if to_node == 12: to_node = 14
        area = self.area * coeffs.area_scale_factors[self.node][to_node]
        delay = self.delay * coeffs.delayfactor(to_node) / coeffs.delayfactor(self.node)
        return Component(to_node, area, delay, self.power)

    def scale_isofreq(self, to_node : int):
        if to_node == 12: to_node = 14
        area = self.area * coeffs.area_scale_factors[self.node][to_node]
        power = self.power * coeffs.powerfactor(to_node) / coeffs.powerfactor(self.node)
        return Component(to_node, area, self.delay, power)
