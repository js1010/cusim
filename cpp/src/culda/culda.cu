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

void CuLDA::LoadModel(float* alpha, float* beta, 
    float* grad_alpha, float* new_beta, int num_words) {
  num_words_ = num_words;
  DEBUG("copy model({} x {})", num_words_, num_topics_);
  dev_alpha_.resize(num_topics_);
  dev_beta_.resize(num_topics_ * num_words_);
  thrust::copy(alpha, alpha + num_topics_, dev_alpha_.begin());
  thrust::copy(beta, beta + num_topics_ * num_words_, dev_beta_.begin());
  alpha_ = alpha; beta_ = beta;
  
  // resize device vector
  grad_alpha_ = grad_alpha;
  new_beta_ = new_beta;
  dev_grad_alpha_.resize(num_topics_ * block_cnt_);
  dev_new_beta_.resize(num_topics_ * num_words_);

  // copy to device
  thrust::copy(grad_alpha_, grad_alpha_ + block_cnt_ * num_topics_, dev_grad_alpha_.begin());
  thrust::copy(new_beta_, new_beta_ + num_words_ * num_topics_, dev_new_beta_.begin());
  dev_gamma_.resize(num_topics_ * block_cnt_);
  dev_new_gamma_.resize(num_topics_ * block_cnt_);
  dev_phi_.resize(num_topics_ * block_cnt_);
  CHECK_CUDA(cudaDeviceSynchronize());
}

std::pair<float, float> CuLDA::FeedData(
    const int* cols, const int* indptr, const bool* vali,
    const int num_cols, const int num_indptr, const int num_iters) {
  
  // copy feed data to GPU memory
  thrust::device_vector<int> dev_cols(num_cols);
  thrust::device_vector<int> dev_indptr(num_indptr + 1);
  thrust::device_vector<bool> dev_vali(num_cols);
  thrust::device_vector<float> dev_train_losses(block_cnt_, 0.0f);
  thrust::device_vector<float> dev_vali_losses(block_cnt_, 0.0f);
  thrust::copy(cols, cols + num_cols, dev_cols.begin());
  thrust::copy(indptr, indptr + num_indptr + 1, dev_indptr.begin());
  thrust::copy(vali, vali + num_cols, dev_vali.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
  DEBUG0("copy feed data to GPU memory");

  // run E step in GPU
  EstepKernel<<<block_cnt_, block_dim_>>>(
    thrust::raw_pointer_cast(dev_cols.data()),
    thrust::raw_pointer_cast(dev_indptr.data()),
    thrust::raw_pointer_cast(dev_vali.data()),
    num_cols, num_indptr, num_topics_, num_iters,
    thrust::raw_pointer_cast(dev_gamma_.data()),
    thrust::raw_pointer_cast(dev_new_gamma_.data()),
    thrust::raw_pointer_cast(dev_phi_.data()),
    thrust::raw_pointer_cast(dev_alpha_.data()),
    thrust::raw_pointer_cast(dev_beta_.data()),
    thrust::raw_pointer_cast(dev_grad_alpha_.data()),
    thrust::raw_pointer_cast(dev_new_beta_.data()),
    thrust::raw_pointer_cast(dev_train_losses.data()),
    thrust::raw_pointer_cast(dev_vali_losses.data()));
  CHECK_CUDA(cudaDeviceSynchronize());
  DEBUG0("run E step in GPU");

  // pull loss
  std::vector<float> train_losses(block_cnt_), vali_losses(block_cnt_);
  thrust::copy(dev_train_losses.begin(), dev_train_losses.end(), train_losses.begin());
  thrust::copy(dev_vali_losses.begin(), dev_vali_losses.end(), vali_losses.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
  DEBUG0("pull loss values");

  // accumulate
  float train_loss = std::accumulate(train_losses.begin(), train_losses.end(), 0.0f);
  float vali_loss = std::accumulate(vali_losses.begin(), vali_losses.end(), 0.0f);
  return {train_loss, vali_loss};
}

void CuLDA::Pull() {
  thrust::copy(dev_grad_alpha_.begin(), dev_grad_alpha_.end(), grad_alpha_);
  thrust::copy(dev_new_beta_.begin(), dev_new_beta_.end(), new_beta_);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void CuLDA::Push() {
  thrust::copy(alpha_, alpha_ + num_topics_, dev_alpha_.begin());
  thrust::copy(grad_alpha_, grad_alpha_ + block_cnt_ * num_topics_, dev_grad_alpha_.begin());
  thrust::copy(beta_, beta_ + num_words_ * num_topics_, dev_beta_.begin());
  thrust::copy(new_beta_, new_beta_ + num_words_ * num_topics_, dev_new_beta_.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
}

int CuLDA::GetBlockCnt() {
  return block_cnt_;
}

}  // namespace cusim
