// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "cuw2v/cuw2v.hpp"
#include "cuw2v/cuda_w2v_kernels.cuh"

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
  INFO("num_dims: {}, block_dim: {}, block_cnt: {}, objective type: {}, neg: {}", 
      num_dims_, block_dim_, block_cnt_, sg_? "skip gram": "cbow", neg_);
  return true;
}


void CuW2V::BuildHuffmanTree(const float* word_count, const int num_words) {
  num_words_ = num_words;
  if (neg_) return;

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
  std::vector<bool> codes;
  std::vector<int> points;
  codes_.clear(); points_.clear();
  codes_.resize(num_words); points_.resize(num_words);
  max_depth_ = 0;
  while (not stack.empty()) {
    std::tie(nodeid, codes, points) = stack.back();
    stack.pop_back();
    if (nodeid < num_words) {
      codes_[nodeid] = codes;
      points_[nodeid] = points;
      max_depth_ = std::max(max_depth_, 
          static_cast<int>(codes.size()));
    } else {
      points.push_back(nodeid - num_words);
      std::vector<bool> left_codes = codes;
      std::vector<bool> right_codes = codes;
      left_codes.push_back(false);
      right_codes.push_back(true);
      auto& node = huffman_nodes[nodeid];
      stack.push_back(make_tuple(node.left, left_codes, points));
      stack.push_back(make_tuple(node.right, right_codes, points));
    }
  }
  
  std::vector<float> host_codes;
  std::vector<int> host_points;
  std::vector<int> host_indptr = {0};
  int size = 0;
  for (int i = 0; i < num_words; ++i) {
    auto& codes = codes_[i];
    auto& points = points_[i];
    int n = codes.size();
    size += n;
    host_indptr.push_back(size);
    for (int j = 0; j < n; ++j) {
      host_codes.push_back(static_cast<float>(codes[j]));
      host_points.push_back(points[j]);
    }
  }
   
  dev_codes_.resize(size); dev_points_.resize(size), dev_indptr_.resize(num_words + 1);
  thrust::copy(host_codes.begin(), host_codes.end(), dev_codes_.begin());
  thrust::copy(host_points.begin(), host_points.end(), dev_points_.begin());
  thrust::copy(host_indptr.begin(), host_indptr.end(), dev_indptr_.begin());
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
  
  // set mutex
  dev_mutex_in_.resize(num_words_);
  dev_mutex_out_.resize(out_words);
  std::vector<int> host_mutex_in(num_words_, 0);
  std::vector<int> host_mutex_out(out_words, 0);
  thrust::copy(host_mutex_in.begin(), host_mutex_in.end(), dev_mutex_in_.begin());
  thrust::copy(host_mutex_out.begin(), host_mutex_out.end(), dev_mutex_out_.begin());
  
  CHECK_CUDA(cudaDeviceSynchronize());
}

int CuW2V::GetBlockCnt() {
  return block_cnt_;
}

}  // namespace cusim
