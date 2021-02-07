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
  return true;
}

void CuLDA::LoadModel(float* alpha, float* beta, int num_words) {
  num_words_ = num_words;
  DEBUG("copy model({} x {})", num_topics_, num_words_);
  dev_alpha_.resize(num_topics_);
  dev_beta_.resize(num_topics_ * num_words_);
  #ifdef HALF_PRECISION
    // conversion to half data and copy
    std::vector<cuda_scalar> halpha(num_topics_), hbeta(num_topics_ * num_words_);
    for (int i = 0; i < num_topics_; ++i) {
      halpha[i] = conversion(alpha[i]);
      for (int j = 0; j < num_words_; ++j) {
        hbeta[i * num_words + j] = conversion(beta[i * num_words + j]);
      }
    }
    thrust::copy(halpha.begin(), halpha.end(), dev_alapha_.begin());
    thrust::copy(hbeta.begin(), hbeta.end(), dev_beta_.begin());
  #else
    thrust::copy(alpha, alpha + num_topics_, dev_alpha_.begin());
    thrust::copy(beta, beta + num_topics_ * num_words_, dev_beta_.begin());
  #endif
  alpha_ = alpha; beta_ = beta;
}

void CuLDA::FeedData(const int* indices, const int* indptr, int num_indices, int num_indptr) {
}

} // namespace cusim
