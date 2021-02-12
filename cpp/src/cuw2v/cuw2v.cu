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
  
  // if zero, we will use hierarchical softmax
  neg_ = opt_["negative_sampling"].int_value(); 
  
  // set seed for constructing random table of negative sampling
  table_seed_ = opt_["table_seed"].int_value();
  const unsigned int table_seed = table_seed_;
  table_rng_.seed(table_seed);

  INFO("num_dims: {}, block_dim: {}, block_cnt: {}, objective type: {}, neg: {}", 
      num_dims_, block_dim_, block_cnt_, sg_? "skip gram": "cbow", neg_);
  return true;
}

void CuW2V::BuildRandomTable(const float* word_count, const int num_words, 
    const int table_size, const int num_threads) {
  num_words_ = num_words;
  table_size_ = table_size;
  std::vector<float> acc;
  float cumsum = 0;
  for (int i = 0; i < num_words; ++i) {
    cumsum += word_count[i];
    acc.push_back(cumsum);
  }

  std::uniform_real_distribution<float> dist(0.0f, cumsum);
  dev_random_table_.resize(table_size_);
  std::vector<int> host_random_table(table_size);
  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < table_size_; ++i) {
      float r = dist(table_rng_);
      int pos = std::lower_bound(acc.begin(), acc.end(), r) - acc.begin();
      host_random_table[i] = pos;
    }
  }

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


float FeedData(const int* cols, const int* indptr, const int num_cols, const int* num_indptr) {
  return 0;
}

}  // namespace cusim
