// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "cuw2v/cuw2v.hpp"
#include "cuw2v/cuda_w2v_kernels.cuh"

namespace cusim {

CuW2V::CuW2V() {
  logger_ = CuSimLogger().get_logger();
  dev_info_ = GetDeviceInfo();
  if (dev_info_.unknown) DEBUG0("Unknown device type");
  INFO("cuda device info, major: {}, minor: {}, multi processors: {}, cores: {}",
       dev_info_.major, dev_info_.minor, dev_info_.mp_cnt, dev_info_.cores);
}

CuW2V::~CuW2V() {}

bool CuW2V::Init(std::string opt_path) {
  std::ifstream in(opt_path.c_str());
  if (not in.is_open()) return false;

  std::string str((std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>());
  std::string err_cmt;
  auto _opt = json11::Json::parse(str, err_cmt);
  if (not err_cmt.empty()) return false;
  opt_ = _opt;
  CuSimLogger().set_log_level(opt_["c_log_level"].int_value());
  num_topics_ = opt_["num_dims"].int_value();
  block_dim_ = opt_["block_dim"].int_value();
  block_cnt_ = opt_["hyper_threads"].number_value() * (dev_info_.cores / block_dim_);
  sg_ = opt_["skip_gram"].bool_value();
  // if zero, we will use hierarchical softmax
  neg_ = opt["negative_sampling"].int_value(); 
  INFO("num_dims: {}, block_dim: {}, block_cnt: {}, objective type: {}, neg: {}", 
      num_dims_, block_dim_, block_cnt_, sg_? "skip gram": "cbow", neg_);
  return true;
}

void CuW2V::LoadModel(float* emb_in, float* emb_out, const int num_words, int num_hs_nodes = 0) {
  num_words_ = num_words;
  out_size_ = neg_? num_words_: num_hs_nodes;
  
  // copy embedding
  DEBUG("copy model({} x {})", num_words_, num_dims_);
  dev_emb_in_.resize(num_words_ * num_dims_);
  dev_emb_out_.resize(out_size_ * num_dims_);
  thrust::copy(emb_in, emb_in + num_words_ * num_dims_, dev_emb_in_.begin());
  thrust::copy(emb_out, emb_out + out_size_ * num_dims_, dev_emb_out_.begin());
  emb_in_ = emb_in; emb_out_ = emb_out;
  
  // set mutex
  dev_mutex_in_.resize(num_words_);
  dev_mutex_out_.resize(out_size_);
  std::vector<int> host_mutex_in(num_words_, 0);
  std::vector<int> host_mutex_out(out_size_, 0);
  thrust::copy(host_mutex_in.begin(), host_mutex_in.end(), dev_mutex_in_.begin());
  thrust::copy(host_mutex_out.begin(), host_mutex_out.end(), dev_mutex_out_.begin());
  
  CHECK_CUDA(cudaDeviceSynchronize());
}

std::pair<float, float> CuLDA::FeedData(
    const int* cols, const int* indptr, const bool* vali,
    const int num_cols, const int num_indptr) {
  
  // copy feed data to GPU memory
  thrust::device_vector<int> dev_cols(num_cols);
  thrust::device_vector<int> dev_indptr(num_indptr + 1);
  thrust::device_vector<float> dev_losses(block_cnt_, 0.0f);
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
    thrust::raw_pointer_cast(dev_vali_losses.data()),
    thrust::raw_pointer_cast(dev_mutex_.data()));
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
