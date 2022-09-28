import dataclasses
from dataclasses import dataclass, fields
import itertools

from ..common import *
from ..arch import *
from .. import model as M

logger = logging.getLogger(__name__)

backward_map = {}

def register_backward(for_class):
    def decorator(x):
        global backward_map
        if for_class not in backward_map:
            backward_map[for_class] = []

        backward_map[for_class].append(x)
        return x
    return decorator

def tuple_compare(a, b):
    if len(a) != len(b): return -1
    score = 0
    for ai, bi in zip(a, b):
        if ai == bi: score += 1
        elif ai is None or bi is None: continue
        elif isinstance(ai, Slice) and bi in ai: continue
        elif isinstance(bi, Slice) and ai in bi: continue
        else: return -1
    return score

def fuzzy_dataclass_match(a, b):
    score = 0

    for f in fields(a):
        ai = getattr(a, f.name)
        bi = getattr(b, f.name)
        if ai is None or bi is None: continue
        elif isinstance(ai, Slice) and bi in ai: score += 1
        elif isinstance(bi, Slice) and ai in bi: score += 1
        elif isinstance(ai, tuple) and isinstance(bi, tuple):
            tscore = tuple_compare(ai, bi)
            if tscore == -1: return False, score
            score += tscore
        elif ai != bi: return False, score
        else: score += 1

    return True, score

def operator(cls):
    def cls_match(self, other):
        return fuzzy_dataclass_match(self, other)

    cls.match = cls_match
    return cls
