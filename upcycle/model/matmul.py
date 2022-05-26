import numpy as np
from dataclasses import dataclass

from .common import *

@dataclass
class MatmulI8MKKN(WorkItem):
    mo : int
    no : int
    k : int
    tm : int
    tn : int

    def __post_init__(self):
        t_issue = 0
        t_read = 0
        t_exec = 0

        for ki in range(0, self.k, 64):
            tk = min(self.k - ki, 64)
            t_setup = np.ceil(self.tm / 2)
            niter = np.ceil(tk / 4)
            load_cyc = np.ceil(tk / 16 / 2)
            init = max(2, self.tn, load_cyc)
            pl = 1 + load_cyc + self.tn

            self._exec_lat = t_setup + (niter - 1) * init + pl
            self._flops = self.tm * self.tn * self.k * 2

    @property
    def exec_lat(self): return self._exec_lat

    @property
    def flops(self): return self._flops

    @property
    def read_trace(self): raise NotImplementedError()

