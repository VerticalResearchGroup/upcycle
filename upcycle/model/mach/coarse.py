from dataclasses import dataclass
from typing import Callable
import numpy as np
import logging
import time

from ...common import *

from ..common import *
from .. import noc

from . import oracle

logger = logging.getLogger(__name__)

@register_sim(CoarseOracle)
def coarse_oracle_sim(arch : Arch, op : Operator, *args, **kwargs):
    return common_sim(
        arch, op, *args, noc_sim_func=oracle.simulate_oracle_noc, **kwargs)
