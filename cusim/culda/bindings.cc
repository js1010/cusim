// Copyright (c) 2021 Jisang Yoon
//  All rights reserved.
//
//  This source code is licensed under the Apache 2.0 license found in the
//  LICENSE file in the root directory of this source tree.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include "culda/culda.hpp"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> float_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> int_array;

class CuLDABind {
 public:
  CuLDABind() {}

  bool Init(std::string opt_path) {
    return obj_.Init(opt_path);
  }

  void LoadModel(py::object& alpha, py::object& beta) {
    float_array _alpha(alpha);
    float_array _beta(beta);
    auto alpha_buffer = _alphpa.request();
    auto beta_buffer = _beta.request();
    if (alpha_buffer.ndim != 1 or beta_buffer.ndim != 2 or
        alpha_buffer.shape[0] != beta_buffer.shape[0]) {
      throw std::runtime_error("invalid alpha or beta");
    }
    int num_words = beta_buffer.shape[1];
    return obj_.LoadModel(_alpha.mutable_data(0), _beta.mutable_data(0), num_words);
  }

 private:
  cusim::CuLDA obj_;
};

PYBIND11_PLUGIN(culda_bind) {
  py::module m("CuLDABind");

  py::class_<IoUtilsBind>(m, "CuLDABind")
  .def(py::init())
  .def("init", &CuLDABind::Init, py::arg("opt_path"))
  .def("load_model", &IoUtilsBind::LoadModel,
      py::arg("alpha"), py::arg("beta"))
  .def("__repr__",
  [](const CuLDABind &a) {
    return "<CuLDABind>";
  }
  );
  return m.ptr();
}
