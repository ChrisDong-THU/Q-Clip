#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "phash_cuda.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_phash_cuda_py, m) {
    m.def("phash",
        &phash_cuda::compute_phash,
        py::arg("paths"),
        py::arg("resize") = 32,
        py::arg("low_freq") = 8,
        py::arg("batch_size") = 128
    );
}
