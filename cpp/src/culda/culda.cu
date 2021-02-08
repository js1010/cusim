// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "culda/culda.hpp"
#include "culda/cuda_lda_kernels.cuh"

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
  InitModel();
}

void CuLDA::InitModel() {
  // resize device vector
  dev_grad_alpha_.resize(num_topics_);
  dev_new_beta_.resize(num_topics_ * num_words_);
  dev_gamma_.resize(num_topics_ * block_cnt_);
  dev_new_gamma_.resize(num_topics_ * block_cnt_);
  dev_phi_.resize(num_topics_ * block_cnt_);
  
  // resize host vector
  grad_alpha_.resize(num_topics_);
  new_beta_.resize(num_topics_ * num_words_);

  // fill zeros
  std::fill(grad_alpha_.begin(), grad_alpha_.end(), 0);
  std::fill(new_beta_.begin(), new_beta_.end(), 0);

  // copy to device
  thrust::copy(grad_alpha_.begin(), grad_alpha_.end(), dev_grad_alpha_.begin());
  thrust::copy(new_beta_.begin(), new_beta_.end(), dev_new_beta_.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void CuLDA::FeedData(const int* indices, const int* indptr, 
    const int num_indices, const int num_indptr, const int num_iters) {
  thrust::device_vector<int> dev_indices(num_indices);
  thrust::device_vector<int> dev_indptr(num_indptr + 1);
  thrust::copy(indices, indices + num_indices, dev_indices.begin());
  thrust::copy(indptr, indptr + num_indptr + 1, dev_indptr.begin());
  CHECK_CUDA(cudaDeviceSynchronize());

  EstepKernel<<<block_cnt_, block_dim_>>>(
    thrust::raw_pointer_cast(dev_indices.data()),
    thrust::raw_pointer_cast(dev_indptr.data()),
    num_indices, num_indptr, num_words_, num_topics_, num_iters,
    thrust::raw_pointer_cast(dev_gamma_.data()),
    thrust::raw_pointer_cast(dev_new_gamma_.data()),
    thrust::raw_pointer_cast(dev_phi_.data()),
    thrust::raw_pointer_cast(dev_alpha_.data()),
    thrust::raw_pointer_cast(dev_beta_.data()),
    thrust::raw_pointer_cast(dev_grad_alpha_.data()),
    thrust::raw_pointer_cast(dev_new_beta_.data()));
  
  CHECK_CUDA(cudaDeviceSynchronize());
}

void CuLDA::Mstep(const int num_docs) {
  thrust::copy(dev_grad_alpha_.begin(), dev_grad_alpha_.end(), grad_alpha_.begin());
  thrust::copy(dev_new_beta_.begin(), dev_new_beta_.end(), new_beta_.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
  

}

} // namespace cusim
