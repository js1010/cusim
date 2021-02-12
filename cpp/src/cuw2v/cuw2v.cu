// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "cuw2v/cuw2v.hpp"
#include "cuw2v/cuda_w2v_base_kernels.cuh"
#include "cuw2v/cuda_w2v_ns_kernels.cuh"
#include "cuw2v/cuda_w2v_hs_kernels.cuh"

namespace cusim {

bool CompareIndex(int lhs, int rhs) {
  return huffman_nodes[lhs].count > huffman_nodes[rhs].count;
}

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
  num_dims_ = opt_["num_dims"].int_value();
  block_dim_ = opt_["block_dim"].int_value();
  block_cnt_ = opt_["hyper_threads"].number_value() * (dev_info_.cores / block_dim_);
  sg_ = opt_["skip_gram"].bool_value();
  use_mean_ = opt_["use_mean"].bool_value();
  window_size_ = opt_["window_size"].int_value();
  lr_ = opt_["lr"].number_value();

  // if zero, we will use hierarchical softmax
  neg_ = opt_["negative_sampling"].int_value(); 
  
  // random seed 
  table_seed_ = opt_["table_seed"].int_value();
  cuda_seed_ = opt_["cuda_seed"].int_value();
  dev_rngs_.resize(block_cnt_);
  InitRngsKernel<<<block_cnt_, 1>>>(
    thrust::raw_pointer_cast(dev_rngs_.data()), cuda_seed_);

  INFO("num_dims: {}, block_dim: {}, block_cnt: {}, objective type: {}, neg: {}", 
      num_dims_, block_dim_, block_cnt_, sg_? "skip gram": "cbow", neg_);
  return true;
}

void CuW2V::BuildRandomTable(const float* word_count, const int num_words, 
    const int table_size, const int num_threads) {
  num_words_ = num_words;
  random_size_ = table_size;
  std::vector<float> acc;
  float cumsum = 0;
  for (int i = 0; i < num_words; ++i) {
    acc.push_back(cumsum);
    cumsum += word_count[i];
  }

  dev_random_table_.resize(random_size_);
  std::vector<int> host_random_table(table_size);
  #pragma omp parallel num_threads(num_threads)
  {
    const unsigned int table_seed = table_seed_ + omp_get_thread_num();
    std::mt19937 rng(table_seed);
    std::uniform_real_distribution<float> dist(0.0f, cumsum);
    #pragma omp for schedule(static)
    for (int i = 0; i < random_size_; ++i) {
      float r = dist(rng);
      int pos = std::lower_bound(acc.begin(), acc.end(), r) - acc.begin();
      host_random_table[i] = pos;
    }
  }
  table_seed_ += num_threads;

  thrust::copy(host_random_table.begin(), host_random_table.end(), dev_random_table_.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void CuW2V::BuildHuffmanTree(const float* word_count, const int num_words) {
  num_words_ = num_words;

  huffman_nodes.clear();
  std::priority_queue<int, std::vector<int>, decltype(&CompareIndex)> pq(CompareIndex);
  for (int i = 0; i < num_words; ++i) {
    huffman_nodes.emplace_back(word_count[i], i, -1, -1);
    pq.push(i);
  }
  for (int i = 0; i < num_words - 1; ++i) {
    auto& min1 = huffman_nodes[pq.top()]; pq.pop();
    auto& min2 = huffman_nodes[pq.top()]; pq.pop();
    huffman_nodes.emplace_back(min1.count + min2.count, i + num_words, min1.index, min2.index);
    pq.push(i + num_words);
  }

  std::vector<std::tuple<int, std::vector<bool>, std::vector<int>>> stack = {{pq.top(), {}, {}}};
  int nodeid;
  std::vector<bool> code;
  std::vector<int> point;
  std::vector<std::vector<bool>> codes(num_words);
  std::vector<std::vector<int>> points(num_words);
  max_depth_ = 0;
  while (not stack.empty()) {
    std::tie(nodeid, code, point) = stack.back();
    stack.pop_back();
    if (nodeid < num_words) {
      codes[nodeid] = code;
      points[nodeid] = point;
      max_depth_ = std::max(max_depth_, 
          static_cast<int>(code.size()));
    } else {
      point.push_back(nodeid - num_words);
      std::vector<bool> left_code = code;
      std::vector<bool> right_code = code;
      left_code.push_back(false);
      right_code.push_back(true);
      auto& node = huffman_nodes[nodeid];
      stack.push_back(make_tuple(node.left, left_code, point));
      stack.push_back(make_tuple(node.right, right_code, point));
    }
  }
  
  std::vector<bool> host_codes;
  std::vector<int> host_points;
  std::vector<int> host_hs_indptr = {0};
  int size = 0;
  for (int i = 0; i < num_words; ++i) {
    code = codes[i];
    point = points[i];
    int n = code.size();
    size += n;
    host_hs_indptr.push_back(size);
    for (int j = 0; j < n; ++j) {
      host_codes.push_back(code[j]);
      host_points.push_back(point[j]);
    }
  }
   
  dev_codes_.resize(size); dev_points_.resize(size), dev_hs_indptr_.resize(num_words + 1);
  thrust::copy(host_codes.begin(), host_codes.end(), dev_codes_.begin());
  thrust::copy(host_points.begin(), host_points.end(), dev_points_.begin());
  thrust::copy(host_hs_indptr.begin(), host_hs_indptr.end(), dev_hs_indptr_.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
  
  huffman_nodes.clear();
}

void CuW2V::LoadModel(float* emb_in, float* emb_out) {
  int out_words = neg_? num_words_: num_words_ - 1;

  // copy embedding
  DEBUG("copy model({} x {})", num_words_, num_dims_);
  dev_emb_in_.resize(num_words_ * num_dims_);
  dev_emb_out_.resize(out_words * num_dims_);
  thrust::copy(emb_in, emb_in + num_words_ * num_dims_, dev_emb_in_.begin());
  thrust::copy(emb_out, emb_out + out_words * num_dims_, dev_emb_out_.begin());
  emb_in_ = emb_in; emb_out_ = emb_out;
  
  CHECK_CUDA(cudaDeviceSynchronize());
}

int CuW2V::GetBlockCnt() {
  return block_cnt_;
}


std::pair<float, float> CuW2V::FeedData(const int* cols, const int* indptr, 
    const int num_cols, const int num_indptr) {
  
  // copy feed data to GPU memory
  thrust::device_vector<int> dev_cols(num_cols); 
  thrust::device_vector<int> dev_indptr(num_indptr + 1);
  thrust::device_vector<float> dev_loss_nume(block_cnt_, 0.0f);
  thrust::device_vector<float> dev_loss_deno(block_cnt_, 0.0f);
  thrust::copy(cols, cols + num_cols, dev_cols.begin());
  thrust::copy(indptr, indptr + num_indptr + 1, dev_indptr.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
  DEBUG0("copy feed data to GPU memory");

  // run GPU kernels
  if (neg_ > 0) {
    if (sg_) {
      W2VNegSgKernel<<<block_cnt_, block_dim_, num_dims_ * sizeof(float)>>>(
        thrust::raw_pointer_cast(dev_cols.data()),
        thrust::raw_pointer_cast(dev_indptr.data()),
        thrust::raw_pointer_cast(dev_random_table_.data()),
        thrust::raw_pointer_cast(dev_rngs_.data()),
        random_size_, num_indptr, num_dims_, neg_, window_size_,
        thrust::raw_pointer_cast(dev_emb_in_.data()),
        thrust::raw_pointer_cast(dev_emb_out_.data()),
        thrust::raw_pointer_cast(dev_loss_nume.data()),
        thrust::raw_pointer_cast(dev_loss_deno.data()),
        lr_);
    } else {
      W2VNegCbowKernel<<<block_cnt_, block_dim_, 2 * num_dims_ * sizeof(float)>>>(
        thrust::raw_pointer_cast(dev_cols.data()),
        thrust::raw_pointer_cast(dev_indptr.data()),
        thrust::raw_pointer_cast(dev_random_table_.data()),
        thrust::raw_pointer_cast(dev_rngs_.data()),
        random_size_, num_indptr, num_dims_, neg_, window_size_,
        thrust::raw_pointer_cast(dev_emb_in_.data()),
        thrust::raw_pointer_cast(dev_emb_out_.data()),
        thrust::raw_pointer_cast(dev_loss_nume.data()),
        thrust::raw_pointer_cast(dev_loss_deno.data()),
        use_mean_, lr_);
    }
  } else {
    if (sg_) {
      W2VHsSgKernel<<<block_cnt_, block_dim_, num_dims_ * sizeof(float)>>>(
        thrust::raw_pointer_cast(dev_cols.data()),
        thrust::raw_pointer_cast(dev_indptr.data()),
        thrust::raw_pointer_cast(dev_codes_.data()),
        thrust::raw_pointer_cast(dev_points_.data()),
        thrust::raw_pointer_cast(dev_hs_indptr_.data()),
        num_indptr, num_dims_, window_size_,
        thrust::raw_pointer_cast(dev_rngs_.data()),
        thrust::raw_pointer_cast(dev_emb_in_.data()),
        thrust::raw_pointer_cast(dev_emb_out_.data()),
        thrust::raw_pointer_cast(dev_loss_nume.data()),
        thrust::raw_pointer_cast(dev_loss_deno.data()),
        lr_);

    } else {
      W2VHsCbowKernel<<<block_cnt_, block_dim_, 2 * num_dims_ * sizeof(float)>>>(
        thrust::raw_pointer_cast(dev_cols.data()),
        thrust::raw_pointer_cast(dev_indptr.data()),
        thrust::raw_pointer_cast(dev_codes_.data()),
        thrust::raw_pointer_cast(dev_points_.data()),
        thrust::raw_pointer_cast(dev_hs_indptr_.data()),
        num_indptr, num_dims_, window_size_,
        thrust::raw_pointer_cast(dev_rngs_.data()),
        thrust::raw_pointer_cast(dev_emb_in_.data()),
        thrust::raw_pointer_cast(dev_emb_out_.data()),
        thrust::raw_pointer_cast(dev_loss_nume.data()),
        thrust::raw_pointer_cast(dev_loss_deno.data()),
        use_mean_, lr_);

    }

  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // accumulate loss nume / deno
  std::vector<float> loss_nume(block_cnt_), loss_deno(block_cnt_);
  thrust::copy(dev_loss_nume.begin(), dev_loss_nume.end(), loss_nume.begin());
  thrust::copy(dev_loss_deno.begin(), dev_loss_deno.end(), loss_nume.begin());
  CHECK_CUDA(cudaDeviceSynchronize());
  float loss_nume_sum = std::accumulate(loss_nume.begin(), loss_nume.end(), 0.0f); 
  float loss_deno_sum = std::accumulate(loss_deno.begin(), loss_deno.end(), 0.0f); 
  DEBUG("loss nume: {}, deno: {}", loss_nume_sum, loss_deno_sum);

  return {loss_nume_sum, loss_deno_sum};
}

void CuW2V::Pull() {
  thrust::copy(dev_emb_in_.begin(), dev_emb_in_.end(), emb_in_);
  thrust::copy(dev_emb_out_.begin(), dev_emb_out_.end(), emb_out_);
  CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace cusim
