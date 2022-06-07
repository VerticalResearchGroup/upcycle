from dataclasses import dataclass
import itertools

from ..common import *
from .. import model as M

backward_map = {}

def register_backward(for_class):
    def decorator(x):
        global backward_map
        if for_class not in backward_map:
            backward_map[for_class] = []

        backward_map[for_class].append(x)
        return x
    return decorator


