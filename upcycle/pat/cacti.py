import os
import tempfile
import subprocess
from dataclasses import dataclass

from .common import *
from . import coeffs

CACTI_DIR = '/research/apps/mcpat/cacti'

@dataclass(frozen=True)
class CactiResult:
    read_j : float
    write_j : float
    leak_w : float
    area_mm2 : float

def cacti(size_bytes, node=7):
    if node == 12: node = 14

    fd, cfgfile = tempfile.mkstemp('.cfg', text=True)
    subprocess.check_call([f'{CACTI_DIR}/gencfg', str(size_bytes), cfgfile ])
    output = subprocess.check_output([f'{CACTI_DIR}/cacti', '-infile', cfgfile])
    os.remove(cfgfile, dir_fd=fd)
    if os.path.exists('out.csv'): os.remove('out.csv')

    for line in output.decode('utf-8').splitlines():
        if 'Total dynamic read energy per access (nJ)' in line:
            read_j = float(line.strip().split()[-1]) * 1e-9
        elif 'Total dynamic write energy per access (nJ)' in line:
            write_j = float(line.strip().split()[-1]) * 1e-9
        elif 'Total leakage power of a bank without power gating, including its network outside (mW)' in line:
            leak_w = float(line.strip().split()[-1]) * 1e-3
        elif 'Cache height x width (mm)' in line:
            height_mm_str, _, width_mm_str = line.strip().split()[-3:]
            height_mm = float(height_mm_str)
            width_mm = float(width_mm_str)

    energy_scale = coeffs.energyfactor(node) / coeffs.energyfactor(32)
    pow_scale = coeffs.powerfactor(node) / coeffs.powerfactor(32)
    area_scale = coeffs.area_scale_factors[32][node]

    return CactiResult(
        read_j * energy_scale,
        write_j * energy_scale,
        leak_w * pow_scale,
        height_mm * width_mm * area_scale)

