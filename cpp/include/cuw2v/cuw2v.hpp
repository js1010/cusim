// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <omp.h>
#include <set>
#include <random>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <queue>
#include <deque>
#include <functional>
#include <vector>
#include <cmath>
#include <chrono> // NOLINT

#include "json11.hpp"
#include "utils/log.hpp"
#include "utils/types.hpp"

namespace cusim {


struct HuffmanTreeNode {
  float count;
  int index, left, right;
  HuffmanTreeNode(float count0, int index0, int left0, int right0) {
    count = count0; index = index0; left = left0; right = right0;
  }
};

std::vector<HuffmanTreeNode> huffman_nodes;
bool CompareIndex(int lhs, int rhs);

class CuW2V {
 public:
  CuW2V();
  ~CuW2V();
  bool Init(std::string opt_path);
  void LoadModel(float* emb_in, float* emb_out);
  void BuildHuffmanTree(const float* word_count, const int num_words);
  int GetBlockCnt();

 private:
  DeviceInfo dev_info_;
  json11::Json opt_;
  std::shared_ptr<spdlog::logger> logger_;
  int block_cnt_, block_dim_;
  int num_dims_, num_words_;
  float *emb_in_, *emb_out_;
  thrust::device_vector<float> dev_emb_in_, dev_emb_out_;

  // variables to construct huffman tree
  int max_depth_;
  std::vector<std::vector<bool>> codes_;
  std::vector<std::vector<int>> points_;
  thrust::device_vector<float> dev_codes_;
  thrust::device_vector<int> dev_points_, dev_indptr_;


  bool sg_;
  int neg_;

  // mutex to handle concurrent model update
  thrust::device_vector<int> dev_mutex_in_, dev_mutex_out_;
};

} // namespace cusim
