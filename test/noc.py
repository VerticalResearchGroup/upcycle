import upcycle as U
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

arch = U.ArchRandomLlc(2e9, 32, 64)
app = U.apps.testapp(U.Dtype.I8, n=512)
print(f'App has {app.flops} flops')

gwl = U.model.place_op('naive', arch, app.oplist[0])


num_steps = max(map(len, gwl.tiles))
cache = [U.model.cache.Cache(128, 16, 6) for _ in range(arch.ntiles)]

total_lat = 0
nocs = []

for step in range(num_steps):
    print(f'Time Step {step}')
    noc = U.model.noc.Noc.from_arch(arch)
    for tid, tile in enumerate(gwl.tiles):
        sr, sc = arch.tile_coords(tid)

        if step >= len(tile): continue

        wi = tile[step]
        for tt in wi.read_trace:
            for l in tt.lines:
                if cache[tid].lookup(l): continue
                cache[tid].insert(l)
                dr, dc = arch.addr_llc_coords(l)
                route = list(noc.get_route((sr, sc), (dr, dc)))
                noc.count_route(route)
                noc.count_route(list(reversed(route)))
                noc.count_route(list(reversed(route)))
                noc.count_route(list(reversed(route)))
                noc.count_route(list(reversed(route)))


    total_lat += noc.latency

    print(f'Latency: {noc.latency}')
    print(f'Effective network bandwidth: {noc.enb} bytes/cyc')
    print(f'Per-core network bandwidth: {noc.enb / arch.ntiles} bytes/cyc/core')
    print(f'Per-core network bandwidth: {noc.enb * arch.freq / arch.ntiles / 1e9} GB/s')

    nocs.append(noc)

print(f'Total Latency: {total_lat} cyc')
print(f'Compute: {app.flops / total_lat} flops/cyc')
print(f'Compute: {app.flops / total_lat / arch.ntiles} flops/cyc/core')
print(f'Efficiency: {app.flops / total_lat / arch.ntiles / 128 * 100} %')

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim([0, 64])
ax.set_ylim([0, 32])
line, = ax.plot([], [])

rects = [[
        patches.Rectangle((c, r), 1, 1, facecolor=(0, 0, 0))
        for c in range(arch.ncols)
    ]
    for r in range(arch.nrows)
]

for r in range(arch.nrows):
    for c in range(arch.ncols):
        ax.add_patch(rects[r][c])

def init(): return line,

def animate(step):
    noc : U.model.noc.Noc = nocs[step]
    max_packets = max([
        noc[r, c].num_out
        for r in range(arch.nrows)
        for c in range(arch.ncols)
    ] + [1])

    patches = []

    for r in range(arch.nrows):
        for c in range(arch.ncols):
            num_in = noc[r, c].num_in
            red = np.clip(num_in / max_packets, 0, 1.0)
            green = 1 - red
            rects[r][c].set_facecolor((red, green, 0))
            patches.append(rects[r][c])

    return patches

anim = FuncAnimation(fig, animate, init_func=init, frames=num_steps, interval=1000, blit=True)
plt.tight_layout()
anim.save('anim.gif')
