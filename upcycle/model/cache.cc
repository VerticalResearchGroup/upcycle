#include <pybind11/pybind11.h>


namespace py = pybind11;


class Cache {
private:
    const size_t nset;
    const size_t nway;
    const size_t laddrbits;

    std::vector<std::vector<std::pair<bool, uint64_t>>> tags;
    std::vector<size_t> mru;

public:
    Cache(size_t _nset, size_t _nway, size_t _laddrbits) :
        nset(_nset), nway(_nway), laddrbits(_laddrbits), tags()
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
        const uint64_t line = (addr >> laddrbits) % nset;
        for (const auto& entry : tags[line]) {
            if (entry.first && entry.second == addr) return true;
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

};


PYBIND11_MODULE(cache, m) {
    py::class_<Cache>(m, "Cache")
        .def(py::init<size_t, size_t, size_t>())
        .def("lookup", &Cache::lookup)
        .def("insert", &Cache::insert);
}

