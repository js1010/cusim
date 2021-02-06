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

  bool Init(std::string opt_path) {
    return obj_.Init(opt_path);
  }

  int LoadStreamFile(std::string filepath) {
    return obj_.LoadStreamFile(filepath);
  }

  std::pair<int, bool> ReadStreamForVocab(int num_lines) {
    return obj_.ReadStreamForVocab(num_lines);
  }

  void GetWordVocab(int min_count) {
    return obj_.GetWordVocab(min_count);
  }

 private:
  cusim::IoUtils obj_;
};

PYBIND11_PLUGIN(ioutils_bind) {
  py::module m("IoUtilsBind");

  py::class_<IoUtilsBind>(m, "IoUtilsBind")
  .def(py::init())
  .def("init", &IoUtilsBind::Init, py::arg("opt_path"))
  .def("load_stream_file", &IoUtilsBind::LoadStreamFile, py::arg("filepath"))
  .def("read_stream_for_vocab", &IoUtilsBind::ReadStreamForVocab, py::arg("num_lines"))
  .def("get_word_vocab", &IoUtilsBind::GetWordVocab, py::arg("min_count"))
  .def("__repr__",
  [](const IoUtilsBind &a) {
    return "<IoUtilsBind>";
  }
  );
  return m.ptr();
}
