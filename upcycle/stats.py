import numpy as np

def cumulative_sum(data : list):
    sums = []
    sums.append(data[0])
    for i in range(1, len(data)):
        sums.append(sums[i - 1] + data[i])
    return sums

def sort_and_csum(data : list, key=None):
    _key = (lambda x: x[1]) if key is None else key
    data = sorted(data, key=_key)
    total = sum([x[0] for x in data])
    xs = cumulative_sum([x[0] / total for x in data])
    ys = [x[1] for x in data]
    return xs, ys

def geo_mean(vals : list):
    return np.power(np.prod(vals), 1 / len(vals))
