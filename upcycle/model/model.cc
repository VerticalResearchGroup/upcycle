#include "model.hh"


PYBIND11_MODULE(c_model, m) {
    py::class_<Cache>(m, "Cache")
        .def(py::init<size_t, size_t, size_t>())
        .def("lookup", &Cache::lookup)
        .def("insert", &Cache::insert)
        .def("get_accesses", &Cache::get_accesses)
        .def("get_hits", &Cache::get_hits)
        .def("reset", &Cache::reset);

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

    py::class_<Slice>(m, "Slice")
        .def(py::init<size_t, size_t, size_t>())
        .def_readonly("start", &Slice::start)
        .def_readonly("stop", &Slice::stop)
        .def_readonly("step", &Slice::step);

    py::class_<AffineTile>(m, "AffineTile")
        .def(py::init<
            size_t,
            const std::vector<size_t>&,
            const std::vector<size_t>&,
            const std::vector<size_t>&
        >())
        .def_readonly("shape", &AffineTile::shape)
        .def_readonly("strides", &AffineTile::strides)
        .def_readonly("idx", &AffineTile::idx);


    m.def("tile_read_trace", &tile_read_trace);
    m.def("oracle_traffic", &oracle_traffic);
    m.def("hier_traffic", &hier_traffic);

}
