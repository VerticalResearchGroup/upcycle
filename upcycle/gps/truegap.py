from dataclasses import dataclass
import sys
from .. import pat
from matplotlib import pyplot as plt

from .chartutils import *

def area_scale(from_node, to_node):
    factors = {
        16: 1.0,
        12: 0.86,
        7: 0.34,
        5: 0.19,
        3: 0.19/1.5,
    }

    return factors[from_node] / factors[to_node]

target_node = int(sys.argv[1])

# 7nm: 0.8 / 0.59
# 5nm: 1.95

freq_speedup = {
    7: 0.8 / 0.59,
    5: 1.95,
    3: 1.95 * 1.11,
}

sbw = 1.0
ac = area_scale(12, target_node)
sc = area_scale(12, target_node) * freq_speedup[target_node]

print(f'sbw: {sbw}')
print(f'sc: {sc}')
print(f'sl: {ac**0.5}')

print('='*20)
for rc in [0.64, 0.5]:
    for rl in [0.1, 0.01]:
        print(f'==== rl={rl} ====')
        rbw = 1 - rc - rl

        speedup = lambda k: 1 / (rbw / sbw + rl / (ac**k) + rc / sc)

        print(f'{rbw:3.2f} & {rc:3.2f} & {speedup(0.5):3.2f} \\\\')

    print()

# with figure(COL_WIDTH, 3, 3, 1, f'truegap-{target_node}', sharex=True) as (fig, axs):
#     axs[0].set_ylabel(f'$\gamma = 1.0$')
#     axs[1].set_ylabel(f'Speedup\n$\gamma = 0.5$')
#     axs[2].set_ylabel(f'$\gamma = 0.25$')
#     axs[2].set_xlabel('$r_c$')

#     xs = np.linspace(0, 1, 1000)

#     plt.xlim([0.2, 0.8])

#     for ax, gamma in zip(axs, [1.0, 0.5, 0.25]):
#         for i, rl in enumerate([0.1, 0.01]):
#             vspeedup = np.vectorize(
#                 lambda rc: 1 / ((1 - rl - rc) / sbw + rl / (sc**gamma) + rc / sc))

#             ax.plot(xs, vspeedup(xs), label=f'$r_l={rl}$', color=colors[i])

#         ax.semilogy()
#         ax.grid()

#     axs[0].legend(loc='upper left', fontsize=8)
#     fig.tight_layout(pad=0.5)



with figure(COL_WIDTH, 2.25, 1, 1, f'truegap-{target_node}', sharex=True) as (fig, ax):
    ax.set_ylabel(f'Speedup')
    ax.set_xlabel('$r_c$')
    xs = np.linspace(0, 1, 1000)

    plt.xlim([0.2, 0.8])
    plt.ylim([1, 15])

    for i, gamma in enumerate([1.0, 0.5, 0.25]):
        for rl in [0.1, 0.01]:
            vspeedup = np.vectorize(
                lambda rc: 1 / ((1 - rl - rc) / sbw + rl / (sc**gamma) + rc / sc))

            linestyle = '-' if rl == 0.1 else '--'
            ax.plot(xs, vspeedup(xs), label=f'$r_l={rl}$, $\gamma={gamma}$', linestyle=linestyle, color=colors[i])

    ax.semilogy()
    ax.grid(which='major', alpha=1.0)
    ax.grid(which='minor', alpha=0.5)

    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout(pad=0.5)
