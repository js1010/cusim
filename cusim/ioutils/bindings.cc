// Copyright (c) 2021 Jisang Yoon
//  All rights reserved.
//
//  This source code is licensed under the Apache 2.0 license found in the
//  LICENSE file in the root directory of this source tree.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include "utils/ioutils.hpp"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> float_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> int_array;

class IoUtilsBind {
 public:
  IoUtilsBind() {}

  bool Init(std::string opt_path) {
    return obj_.Init(opt_path);
  }

  int64_t LoadStreamFile(std::string filepath) {
    return obj_.LoadStreamFile(filepath);
  }

  std::pair<int, int> ReadStreamForVocab(int num_lines, int num_threads) {
    return obj_.ReadStreamForVocab(num_lines, num_threads);
  }

  std::pair<int, int> TokenizeStream(int num_lines, int num_threads) {
    return obj_.TokenizeStream(num_lines, num_threads);
  }

  void GetWordVocab(int min_count, std::string keys_path, std::string count_path) {
    obj_.GetWordVocab(min_count, keys_path, count_path);
  }

  void GetToken(py::object& rows, py::object& cols, py::object& indptr) {
    int_array _rows(rows);
    int_array _cols(cols);
    int_array _indptr(indptr);
    obj_.GetToken(_rows.mutable_data(0), _cols.mutable_data(0), _indptr.mutable_data(0));
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
  .def("read_stream_for_vocab", &IoUtilsBind::ReadStreamForVocab,
      py::arg("num_lines"), py::arg("num_threads"))
  .def("tokenize_stream", &IoUtilsBind::TokenizeStream,
      py::arg("num_lines"), py::arg("num_threads"))
  .def("get_word_vocab", &IoUtilsBind::GetWordVocab,
      py::arg("min_count"), py::arg("keys_path"), py::arg("count_path"))
  .def("get_token", &IoUtilsBind::GetToken,
      py::arg("indices"), py::arg("indptr"), py::arg("offset"))
  .def("__repr__",
  [](const IoUtilsBind &a) {
    return "<IoUtilsBind>";
  }
  );
  return m.ptr();
}
