import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

from . import model as M

def soc_animate(self : M.Soc, fig, ax) -> FuncAnimation:

    line, = ax.plot([], [])
    rects = [[
            patches.Rectangle((c, r), 1, 1, facecolor=(0, 0, 0))
            for c in range(self.arch.ncols)
        ]
        for r in range(self.arch.nrows)
    ]

    for r in range(self.arch.nrows):
        for c in range(self.arch.ncols):
            ax.add_patch(rects[r][c])

    def init(): return line,

    def animate(step):
        noc : M.noc.Noc = self.noc(step)
        max_packets = max([
            noc[r, c].num_out
            for r in range(self.arch.nrows)
            for c in range(self.arch.ncols)
        ] + [1])

        patches = []

        for r in range(self.arch.nrows):
            for c in range(self.arch.ncols):
                num_in = noc[r, c].num_in
                red = np.clip(num_in / max_packets, 0, 1.0)
                green = 1 - red
                rects[r][c].set_facecolor((red, green, 0))
                patches.append(rects[r][c])

        return patches

    return FuncAnimation(fig, animate, init_func=init, frames=self.nsteps, interval=500, blit=True)

M.Soc.animate = soc_animate
