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


// reference: https://people.math.sc.edu/Burkardt/cpp_src/asa121/asa121.cpp
inline float Trigamma(float x) {
  const float a = 0.0001f;
  const float b = 5.0f;
  const float b2 =  0.1666666667f;
  const float b4 = -0.03333333333f;
  const float b6 =  0.02380952381f;
  const float b8 = -0.03333333333f;
  float value = 0, y = 0, z = x;
  if (x <= a) return 1.0f / x / x;
  while (z < b) {
    value += 1.0f / z / z;
    z++;
  }
  y = 1.0f / z / z;
  value += value + 0.5 * y + (1.0
    + y * (b2
    + y * (b4
    + y * (b6
    + y * b8)))) / z;
  return value;
}


class CuLDA {
 public:
  CuLDA();
  ~CuLDA();
  bool Init(std::string opt_path);
  void LoadModel(float* alpha, float* beta,
      float* grad_alpha, float* new_beta, const int num_words);
  std::pair<float, float> FeedData(
      const int* indices, const int* indptr,
      const bool* vali, const float* counts,
      float* gamma, const bool init_gamma,
      const int num_indices, const int num_indptr,
      const int num_iters);
  void Pull();
  void Push();
  int GetBlockCnt();

 private:
  DeviceInfo dev_info_;
  json11::Json opt_;
  std::shared_ptr<spdlog::logger> logger_;
  std::unique_ptr<CuSimLogger> logger_container_;
  thrust::device_vector<float> dev_alpha_, dev_beta_;
  thrust::device_vector<float> dev_grad_alpha_, dev_new_beta_;
  thrust::device_vector<int> dev_locks_;

  float *alpha_, *beta_, *grad_alpha_, *new_beta_;
  int block_cnt_, block_dim_;
  int num_topics_, num_words_;
};

} // namespace cusim
