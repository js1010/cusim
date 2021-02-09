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

  void LoadModel(py::object& alpha, py::object& beta,
      py::object& grad_alpha, py::object& new_beta) {
    // check shape of alpha and beta
    float_array _alpha(alpha);
    float_array _beta(beta);
    auto alpha_buffer = _alpha.request();
    auto beta_buffer = _beta.request();
    if (alpha_buffer.ndim != 1 or beta_buffer.ndim != 2 or
        alpha_buffer.shape[0] != beta_buffer.shape[1]) {
      throw std::runtime_error("invalid alpha or beta");
    }

    // check shape of grad alpha and new beta
    float_array _grad_alpha(grad_alpha);
    float_array _new_beta(new_beta);
    auto grad_alpha_buffer = _grad_alpha.request();
    auto new_beta_buffer = _new_beta.request();
    if (grad_alpha_buffer.ndim != 1 or
        new_beta_buffer.ndim != 2 or
        grad_alpha_buffer.shape[0] != new_beta_buffer.shape[1]) {
      throw std::runtime_error("invalid grad_alpha or new_beta");
    }

    int num_words = beta_buffer.shape[1];

    return obj_.LoadModel(_alpha.mutable_data(0),
        _beta.mutable_data(0),
        _grad_alpha.mutable_data(0),
        _new_beta.mutable_data(0), num_words);
  }

  void FeedData(py::object& indices, py::object indptr, const int num_iters) {
    int_array _indices(indices);
    int_array _indptr(indptr);
    auto indices_buffer = _indices.request();
    auto indptr_buffer = _indptr.request();
    if (indices_buffer.ndim != 1 or indptr_buffer.ndim != 1) {
      throw std::runtime_error("invalid indices or indptr");
    }
    int num_indices = indices_buffer.shape[0];
    int num_indptr = indptr_buffer.shape[0];
    obj_.FeedData(_indices.data(0), _indptr.data(0), num_indices, num_indptr, num_iters);
  }

  void Pull() {
    obj_.Pull();
  }

  void Push() {
    obj_.Push();
  }

 private:
  cusim::CuLDA obj_;
};

PYBIND11_PLUGIN(culda_bind) {
  py::module m("CuLDABind");

  py::class_<CuLDABind>(m, "CuLDABind")
  .def(py::init())
  .def("init", &CuLDABind::Init, py::arg("opt_path"))
  .def("load_model", &CuLDABind::LoadModel,
      py::arg("alpha"), py::arg("beta"),
      py::arg("grad_alpha"), py::arg("new_beta"))
  .def("feed_data", &CuLDABind::FeedData,
      py::arg("indices"), py::arg("indptr"), py::arg("num_iters"))
  .def("pull", &CuLDABind::Pull)
  .def("push", &CuLDABind::Push)
  .def("__repr__",
  [](const CuLDABind &a) {
    return "<CuLDABind>";
  }
  );
  return m.ptr();
}
