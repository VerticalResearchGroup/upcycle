#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

class Cache {
public:
    const size_t nset;
    const size_t nway;
    const size_t laddrbits;

private:
    size_t accesses;
    size_t hits;

    std::vector<std::vector<std::pair<bool, uint64_t>>> tags;
    std::vector<size_t> mru;

public:
    Cache(size_t _nset, size_t _nway, size_t _laddrbits) :
        nset(_nset), nway(_nway), laddrbits(_laddrbits), accesses(0), hits(0), tags()
    {
        for (size_t set = 0; set < nset; set++) {
            tags.push_back({});
            mru.push_back(0);
            for (size_t way = 0; way < nway; way++) {
                tags[set].push_back({false, 0});
            }
        }
    }

    inline bool lookup(uint64_t addr) {
        accesses++;
        const uint64_t line = (addr >> laddrbits) % nset;
        for (const auto& entry : tags[line]) {
            if (entry.first && entry.second == addr) {
                hits++;
                return true;
            }
        }
        return false;
    }

    inline void insert(uint64_t addr) {
        if (lookup(addr)) return;
        const uint64_t setid = (addr >> laddrbits) % nset;
        auto& set = tags[setid];

        for (size_t way = 0; way < nway; way++) {
            auto& entry = set[way];
            if (!entry.first) {
                entry.first = true;
                entry.second = addr;
                mru[setid] = way;
                return;
            }
        }

        size_t way = (mru[setid] + 1) % nway;
        auto& entry = set[way];
        entry.first = true;
        entry.second = addr;
        mru[setid] = way;
    }

    inline void reset() {
        accesses = 0;
        hits = 0;
    }

    inline size_t get_accesses() const { return accesses; }
    inline size_t get_hits() const { return hits; }

};

struct Slice {
    size_t start;
    size_t stop;
    size_t step;
};

struct AffineTile {
    size_t tid;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<Slice> idx;
};

struct TileMask {
    uint64_t mask[32] = {0};

    void set(uint64_t tid) {
        assert(tid < 2048 && "tid out of range");
        mask[tid >> 6] |= 1ull << (tid & 63);
    }

    std::vector<uint64_t> tiles() const {
        std::vector<uint64_t> ids;
        for (uint64_t tid = 0; tid < 2048; tid++) {
            if (mask[tid >> 6] & (1ull << (tid & 63))) {
                ids.push_back(tid);
            }
        }
        return ids;
    }
};

struct DestList {
    std::map<uint64_t, TileMask> dests;
    void set(uint64_t addr, uint64_t tid) { dests[addr].set(tid); }
    std::vector<uint64_t> tiles(uint64_t addr) { return dests[addr].tiles(); }
};

#define READ_INNER(cur, l1, dl, tid, addr) \
    const uint64_t line = (addr >> l1.laddrbits) << l1.laddrbits; \
    if (line == cur) continue; \
    cur = line; \
    const bool hit = l1.lookup(cur); \
    l1.insert(cur); \
    if (!hit) dl.set(addr, tid);

void tile_read_trace1(Cache& l1, DestList& dl, AffineTile& at, uint64_t tid) {
    assert(at.shape.size() == 1 && "AffineTile must be 1D");

    const uint64_t upper = at.tid << 32;
    uint64_t cur = (uint64_t)-1;

    const auto& i0_slice = at.idx[0];
    for (size_t i0 = i0_slice.start; i0 < i0_slice.stop; i0 += i0_slice.step) {
        const uint64_t i0_base = i0 * at.strides[0];
        const uint64_t addr = upper | i0_base;
        READ_INNER(cur, l1, dl, tid, addr)
    }
}

void tile_read_trace2(Cache& l1, DestList& dl, AffineTile& at, uint64_t tid) {
    assert(at.shape.size() == 2 && "AffineTile must be 2D");

    const uint64_t upper = at.tid << 32;
    uint64_t cur = (uint64_t)-1;

    const auto& i0_slice = at.idx[0];
    for (size_t i0 = i0_slice.start; i0 < i0_slice.stop; i0 += i0_slice.step) {
        const uint64_t i0_base = i0 * at.strides[0];
        const auto& i1_slice = at.idx[1];
        for (size_t i1 = i1_slice.start; i1 < i1_slice.stop; i1 += i1_slice.step) {
            const uint64_t i1_base = i1 * at.strides[1];
            const uint64_t addr = upper | (i0_base + i1_base);
            READ_INNER(cur, l1, dl, tid, addr)
        }
    }
}

void tile_read_trace3(Cache& l1, DestList& dl, AffineTile& at, uint64_t tid) {
    assert(at.shape.size() == 3 && "AffineTile must be 3D");

    const uint64_t upper = at.tid << 32;
    uint64_t cur = (uint64_t)-1;

    const auto& i0_slice = at.idx[0];
    for (size_t i0 = i0_slice.start; i0 < i0_slice.stop; i0 += i0_slice.step) {
        const uint64_t i0_base = i0 * at.strides[0];
        const auto& i1_slice = at.idx[1];
        for (size_t i1 = i1_slice.start; i1 < i1_slice.stop; i1 += i1_slice.step) {
            const uint64_t i1_base = i1 * at.strides[1];
            const auto& i2_slice = at.idx[2];
            for (size_t i2 = i2_slice.start; i2 < i2_slice.stop; i2 += i2_slice.step) {
                const uint64_t i2_base = i2 * at.strides[2];
                const uint64_t addr = upper | (i0_base + i1_base + i2_base);
                READ_INNER(cur, l1, dl, tid, addr)
            }
        }
    }
}

void tile_read_trace4(Cache& l1, DestList& dl, AffineTile& at, uint64_t tid) {
    assert(at.shape.size() == 4 && "AffineTile must be 4D");

    const uint64_t upper = at.tid << 32;
    uint64_t cur = (uint64_t)-1;

    const auto& i0_slice = at.idx[0];
    for (size_t i0 = i0_slice.start; i0 < i0_slice.stop; i0 += i0_slice.step) {
        const uint64_t i0_base = i0 * at.strides[0];
        const auto& i1_slice = at.idx[1];
        for (size_t i1 = i1_slice.start; i1 < i1_slice.stop; i1 += i1_slice.step) {
            const uint64_t i1_base = i1 * at.strides[1];
            const auto& i2_slice = at.idx[2];
            for (size_t i2 = i2_slice.start; i2 < i2_slice.stop; i2 += i2_slice.step) {
                const uint64_t i2_base = i2 * at.strides[2];
                const auto& i3_slice = at.idx[3];
                for (size_t i3 = i3_slice.start; i3 < i3_slice.stop; i3 += i3_slice.step) {
                    const uint64_t i3_base = i3 * at.strides[3];
                    const uint64_t addr = upper | (i0_base + i1_base + i2_base + i3_base);
                    READ_INNER(cur, l1, dl, tid, addr)
                }
            }
        }
    }
}

void tile_read_trace5(Cache& l1, DestList& dl, AffineTile& at, uint64_t tid) {
    assert(at.shape.size() == 5 && "AffineTile must be 5D");

    const uint64_t upper = at.tid << 32;
    uint64_t cur = (uint64_t)-1;

    const auto& i0_slice = at.idx[0];
    for (size_t i0 = i0_slice.start; i0 < i0_slice.stop; i0 += i0_slice.step) {
        const uint64_t i0_base = i0 * at.strides[0];
        const auto& i1_slice = at.idx[1];
        for (size_t i1 = i1_slice.start; i1 < i1_slice.stop; i1 += i1_slice.step) {
            const uint64_t i1_base = i1 * at.strides[1];
            const auto& i2_slice = at.idx[2];
            for (size_t i2 = i2_slice.start; i2 < i2_slice.stop; i2 += i2_slice.step) {
                const uint64_t i2_base = i2 * at.strides[2];
                const auto& i3_slice = at.idx[3];
                for (size_t i3 = i3_slice.start; i3 < i3_slice.stop; i3 += i3_slice.step) {
                    const uint64_t i3_base = i3 * at.strides[3];
                    const auto& i4_slice = at.idx[4];
                    for (size_t i4 = i4_slice.start; i4 < i4_slice.stop; i4 += i4_slice.step) {
                        const uint64_t i4_base = i4 * at.strides[4];
                        const uint64_t addr = upper | (i0_base + i1_base + i2_base + i3_base + i4_base);
                        READ_INNER(cur, l1, dl, tid, addr)
                    }
                }
            }
        }
    }
}

void tile_read_trace(Cache& l1, DestList& dl, AffineTile& at, uint64_t tid) {
    switch (at.shape.size()) {
        case 1: tile_read_trace1(l1, dl, at, tid); break;
        case 2: tile_read_trace2(l1, dl, at, tid); break;
        case 3: tile_read_trace3(l1, dl, at, tid); break;
        case 4: tile_read_trace4(l1, dl, at, tid); break;
        case 5: tile_read_trace5(l1, dl, at, tid); break;
        default: assert(false && "AffineTile must be 1D-5D");
    }
}

enum NocDir {
    OUT_NORTH = 0,
    OUT_SOUTH = 1,
    OUT_EAST = 2,
    OUT_WEST = 3,
    INJECT = 4,
    EJECT = 5,
    DIRMAX = 6
};

inline void get_hops(
    std::pair<size_t, size_t> src,
    std::pair<size_t, size_t> dst,
    std::set<std::tuple<size_t, size_t, size_t>>& hops)
{
    hops.insert({src.first, src.second, INJECT});
    hops.insert({dst.first, dst.second, EJECT});

    while (src.second < dst.second) {
        hops.insert({src.first, src.second, OUT_EAST});
        src.second += 1;
    }

    while (src.second > dst.second) {
        hops.insert({src.first, src.second, OUT_WEST});
        src.second -= 1;
    }

    while (src.first < dst.first) {
        hops.insert({src.first, src.second, OUT_NORTH});
        src.first += 1;
    }

    while (src.first > dst.first) {
        hops.insert({src.first, src.second, OUT_SOUTH});
        src.first -= 1;
    }

}

template<uint64_t LB, uint64_t NR, uint64_t NC>
void _oracle_traffic(const DestList& dl, py::array_t<uint64_t> traffic) {
    const size_t nrows = traffic.shape(0);
    const size_t ncols = traffic.shape(1);
    const size_t ndirs = traffic.shape(2);
    assert(nrows == NR);
    assert(ncols == NC);
    assert(ndirs == DIRMAX);

    for (const auto& kv : dl.dests) {
        const auto& addr = kv.first;
        const auto& mask = kv.second;
        const size_t line = addr >> LB;
        const size_t src_tid = line & ((NR * NC) - 1);

        std::set<std::tuple<size_t, size_t, size_t>> hops;
        std::pair<size_t, size_t> src = {(line / NC) % NR, line % NC};
        assert(src.first < NR);
        assert(src.second < NC);

        for (const auto& tid : mask.tiles()) {
            std::pair<size_t, size_t> dst = {(line / NC) % NR, line % NC};
            assert(dst.first < NR);
            assert(dst.second < NC);
            get_hops(src, dst, hops);
        }

        for (const auto& hop : hops) {
            traffic.mutable_at(std::get<0>(hop), std::get<1>(hop), std::get<2>(hop)) += 1;
        }
    }

}


void oracle_traffic(
    uint64_t lbits,
    uint64_t nrows,
    uint64_t ncols,
    const DestList& dl,
    py::array_t<uint64_t> traffic)
{
#define SUPPORT_ARCH(lb, nr, nc) \
    case (nr << 16) | nc: \
        _oracle_traffic<lb, nr, nc>(dl, traffic); \
        break;

    uint64_t packed_tile_spec = (nrows << 16) | ncols;
    switch (lbits) {
    case 5:
        switch (packed_tile_spec) {
            SUPPORT_ARCH(5, 32, 64);
        }
        break;
    case 6:
        switch (packed_tile_spec) {
            SUPPORT_ARCH(6, 32, 64);
        }
        break;
    }
}

