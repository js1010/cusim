// Copyright (c) 2021 Jisang Yoon
//  All rights reserved.
//
//  This source code is licensed under the Apache 2.0 license found in the
//  LICENSE file in the root directory of this source tree.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include "cuw2v/cuw2v.hpp"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> float_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> int_array;

class CuW2VBind {
 public:
  CuW2VBind() {}

  bool Init(std::string opt_path) {
    return obj_.Init(opt_path);
  }

  void LoadModel(py::object& emb_in, py::object& emb_out) {
    // check shape of alpha and beta
    float_array _emb_in(emb_in);
    float_array _emb_out(emb_out);
    auto emb_in_buffer = _emb_in.request();
    auto emb_out_buffer = _emb_out.request();
    if (emb_in_buffer.ndim != 2 or emb_out_buffer.ndim != 2 or
        emb_in_buffer.shape[1] != emb_out_buffer.shape[1]) {
      throw std::runtime_error("invalid emb_in or emb_out");
    }

    return obj_.LoadModel(_emb_in.mutable_data(0), _emb_out.mutable_data(0));
  }

  void BuildRandomTable(py::object& word_count, int table_size, int num_threads) {
    float_array _word_count(word_count);
    auto wc_buffer = _word_count.request();
    if (wc_buffer.ndim != 1) {
      throw std::runtime_error("invalid word count");
    }
    int num_words = wc_buffer.shape[0];
    obj_.BuildRandomTable(_word_count.data(0), num_words, table_size, num_threads);
  }

  void BuildHuffmanTree(py::object& word_count) {
    float_array _word_count(word_count);
    auto wc_buffer = _word_count.request();
    if (wc_buffer.ndim != 1) {
      throw std::runtime_error("invalid word count");
    }
    int num_words = wc_buffer.shape[0];
    obj_.BuildHuffmanTree(_word_count.data(0), num_words);
  }

  std::pair<float, float> FeedData(py::object& cols, py::object& indptr) {
    int_array _cols(cols);
    int_array _indptr(indptr);
    auto cols_buffer = _cols.request();
    auto indptr_buffer = _indptr.request();
    if (cols_buffer.ndim != 1 or indptr_buffer.ndim != 1) {
      throw std::runtime_error("invalid cols or indptr");
    }
    int num_cols = cols_buffer.shape[0];
    int num_indptr = indptr_buffer.shape[0] - 1;
    return obj_.FeedData(_cols.data(0), _indptr.data(0), num_cols, num_indptr);
  }

  void Pull() {
    obj_.Pull();
  }

  int GetBlockCnt() {
    return obj_.GetBlockCnt();
  }

 private:
  cusim::CuW2V obj_;
};

PYBIND11_PLUGIN(cuw2v_bind) {
  py::module m("CuW2VBind");

  py::class_<CuW2VBind>(m, "CuW2VBind")
  .def(py::init())
  .def("init", &CuW2VBind::Init, py::arg("opt_path"))
  .def("load_model", &CuW2VBind::LoadModel,
      py::arg("emb_in"), py::arg("emb_out"))
  .def("feed_data", &CuW2VBind::FeedData,
      py::arg("cols"), py::arg("indptr"))
  .def("pull", &CuW2VBind::Pull)
  .def("build_random_table", &CuW2VBind::BuildRandomTable,
      py::arg("word_count"), py::arg("table_size"), py::arg("num_threads"))
  .def("build_huffman_tree", &CuW2VBind::BuildHuffmanTree,
      py::arg("word_count"))
  .def("get_block_cnt", &CuW2VBind::GetBlockCnt)
  .def("__repr__",
  [](const CuW2VBind &a) {
    return "<CuW2VBind>";
  }
  );
  return m.ptr();
}
