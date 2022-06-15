import dataclasses
from dataclasses import dataclass, fields
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


def fuzzy_dataclass_match(a, b):
    score = 0

    for f in fields(a):
        ai = getattr(a, f.name)
        bi = getattr(b, f.name)
        if ai is None or bi is None: continue
        elif isinstance(ai, Slice) and bi in ai: score += 1
        elif isinstance(bi, Slice) and ai in bi: score += 1
        elif ai != bi: return False, score
        else: score += 1

    return True, score

def operator(cls):
    def cls_match(self, other):
        return fuzzy_dataclass_match(self, other)

    cls.match = cls_match
    return cls


pg_placement_map = {}

def placement_profile(archclasses, opprof):
    def decorator(f):
        global pg_placement_map
        if isinstance(archclasses, list):
            for archclass in archclasses:
                key = (archclass, type(opprof))
                if key not in pg_placement_map: pg_placement_map[key] = []
                pg_placement_map[key].append((opprof, f))

        else:
            key = (archclasses, type(opprof))
            if key not in pg_placement_map: pg_placement_map[key] = []
            pg_placement_map[key].append((opprof, f))

    return decorator

def profiled_placement(arch, op, fallback):
    global pg_placement_map
    key = (type(arch), type(op))
    if key not in pg_placement_map:
        logger.debug(f'No placement profiles for {key}. Using Fallback...')
        return fallback(arch, op)

    best_score, best_func = -1, None

    logger.debug(f'Looking for placement profile for {op}')
    for prof, f in pg_placement_map[key]:
        valid, score = prof.match(op)
        logger.debug(f'    + {prof} valid={valid} score={score} (best={best_score})')
        if valid and (score > best_score):
            best_score = score
            best_func = f


    if best_func is None:
        logger.debug(f'No valid profile for {op}. Using Fallback...')
        return fallback(arch, op)

    logger.debug(f'Profiled placement: {op} -> {best_func.__name__} (score = {best_score})')

    return best_func(arch, op)
