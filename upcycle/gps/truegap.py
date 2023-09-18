from dataclasses import dataclass
from .. import pat

def area_scale(from_node, to_node):
    factors = {
        16: 1.0,
        12: 0.86,
        7: 0.34,
        5: 0.19,
        3: 0.19/1.5,
    }

    return factors[from_node] / factors[to_node]

@dataclass
class GalileoConfig:
    flops : float
    area : float
    ns_area : float

    @property
    def s_area(self): return self.area - self.ns_area

def low_ami_speedup(from_cfg : GalileoConfig, to_cfg : GalileoConfig, k=0.2):
    return (to_cfg.s_area / from_cfg.s_area) ** k

galileos = {
    12: GalileoConfig(flops=120e12, area=195, ns_area=50),
    # 7:  GalileoConfig(flops=262e12, area=417, ns_area=100),
    # 5:  GalileoConfig(flops=715e12, area=834, ns_area=100),
    3:  GalileoConfig(flops=555e12, area=830, ns_area=100),
}


sbw = 1.0
sc = 555 / 120

print('='*20)
for rl in [0.01, 0.1]:
    print(f'==== rl={rl} ====')
    for rbw_ in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rbw = rbw_ - rl
        rc = 1 - rbw - rl

        speedup = lambda k: 1 / (rbw / sbw + rl / (sc**k) + rc / sc)

        print(f'{rbw:3.2f} & {rc:3.2f} & {speedup(0.25):3.2f} & {speedup(0.5):3.2f} & {speedup(1.0):3.2f} \\\\')

    print()

