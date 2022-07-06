#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

struct TileMask {
    uint64_t mask[32] = {0};

    void set(uint64_t tid) {
        assert(tid < 2048 && "tid out of range");
        mask[tid >> 6] |= 1ull << (tid & 63);
    }

    std::vector<uint64_t> tiles() {
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

    void set(uint64_t addr, uint64_t tid) {
        dests[addr].set(tid);
    }

    std::vector<uint64_t> tiles(uint64_t addr) {
        return dests[addr].tiles();
    }
};

PYBIND11_MODULE(destlist, m) {
    py::class_<TileMask>(m, "TileMask")
        .def(py::init<>())
        .def("set", &TileMask::set)
        .def("tiles", &TileMask::tiles);

    py::class_<DestList>(m, "DestList")
        .def(py::init<>())
        .def("set", &DestList::set)
        .def("tiles", &DestList::tiles)
        .def_readonly("dests", &DestList::dests)
        .def("__len__", [] (const DestList& d) { return d.dests.size(); });
}

