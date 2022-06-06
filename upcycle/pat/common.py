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
        self.area *= coeffs.area_scale_factors[self.node][to_node]
        self.delay *= coeffs.delayfactor(to_node) / coeffs.delayfactor(self.node)
        self.node = to_node
        return self

    def scale_isofreq(self, to_node : int):
        self.area *= coeffs.area_scale_factors[self.node][to_node]
        self.power *= coeffs.powerfactor(to_node) / coeffs.powerfactor(self.node)
        self.node = to_node
        return self
