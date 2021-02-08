// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "culda/culda.hpp"

namespace cusim {

CuLDA::CuLDA() {
  logger_ = CuSimLogger().get_logger();
  dev_info_ = GetDeviceInfo();
  if (dev_info_.unknown) DEBUG0("Unknown device type");
  INFO("cuda device info, major: {}, minor: {}, multi processors: {}, cores: {}",
       dev_info_.major, dev_info_.minor, dev_info_.mp_cnt, dev_info_.cores);
}

CuLDA::~CuLDA() {}

bool CuLDA::Init(std::string opt_path) {
  std::ifstream in(opt_path.c_str());
  if (not in.is_open()) return false;

  std::string str((std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>());
  std::string err_cmt;
  auto _opt = json11::Json::parse(str, err_cmt);
  if (not err_cmt.empty()) return false;
  opt_ = _opt;
  CuSimLogger().set_log_level(opt_["c_log_level"].int_value());
  num_topics_ = opt_["num_topics"].int_value();
  block_dim_ = opt_["block_dim"].int_value();
  block_cnt_ = opt_["hyper_threads"].number_value() * (dev_info_.cores / block_dim_);
  INFO("num_topics: {}, block_dim: {}, block_cnt: {}", num_topics_, block_dim_, block_cnt_);
  return true;
}

void CuLDA::LoadModel(float* alpha, float* beta, int num_words) {
  num_words_ = num_words;
  DEBUG("copy model({} x {})", num_topics_, num_words_);
  dev_alpha_.resize(num_topics_);
  dev_beta_.resize(num_topics_ * num_words_);
  thrust::copy(alpha, alpha + num_topics_, dev_alpha_.begin());
  thrust::copy(beta, beta + num_topics_ * num_words_, dev_beta_.begin());
  alpha_ = alpha; beta_ = beta;
}

void CuLDA::FeedData(const int* indices, const int* indptr, 
    int num_indices, int num_indptr) {
  thrust::device_vector<int> dev_phi(num_indices * num_topics_);
  thrust::device_vector<int> dev_gamma(num_indptr * num_topics_);
}

} // namespace cusim
