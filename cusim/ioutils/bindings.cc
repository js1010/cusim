// Copyright (c) 2021 Jisang Yoon
//  All rights reserved.
//
//  This source code is licensed under the Apache 2.0 license found in the
//  LICENSE file in the root directory of this source tree.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include "ioutils.hpp"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> float_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> int_array;

class IoUtilsBind {
 public:
  IoUtilsBind() {}
  LoadGensimVocab(std::string filepath, int min_count) {
    obj_.LoadGensimVocab(filepath, min_count);
  }
 private:
  cusim::IoUtils obj_;
};

PYBIND11_PLUGIN(ioutils_bind) {
  py::module m("IoUtilsBind");

  py::class_<IoUtilsBind>(m, "IoUtilsBind")
  .def(py::init())
  .def("load_gensim_vocab", &IoUtilsBind::Init, py::arg("filepath"), py::arg("min_count"))
  .def("__repr__",
  [](const IoUtilsBind &a) {
    return "<IoUtilsBind>";
  }
  );
  return m.ptr();
}
