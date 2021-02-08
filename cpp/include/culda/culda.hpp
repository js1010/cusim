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
#include "utils/cuda_utils_kernels.cuh"

namespace cusim {

class CuLDA {
 public:
  CuLDA();
  ~CuLDA();
  bool Init(std::string opt_path);
  void LoadModel(float* alpha, float* beta, int num_words);
  void FeedData(const int* indices, const int* indptr,
      int num_indices, int num_indptr);

 private:
  DeviceInfo dev_info_;
  json11::Json opt_;
  std::shared_ptr<spdlog::logger> logger_;
  thrust::device_vector<cuda_scalar> dev_alpha_, dev_beta_;
  const float *alpha_, *beta_;
  int block_cnt_, block_dim_;
  int num_topics_, num_words_;
};

} // namespace cusim
