// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "cuw2v/cuw2v.hpp"

namespace cusim {

struct PqItem {
  float count;
  int index;
  PqItem *left, *right;
  bool operator <(const PqItem& left, const PqItem& right) {
    return std::tie(left.count, left.index) < std::tie(right.count, right.index);
  }
}

int CuW2V::BuildHuffmanTree(const float* word_count, const int num_words) {
  num_words_ = num_words;
  if (neg_) {
    out_size_ = num_words_;
    return;
  }
  std::priority_queue<PqItem> pq;
  for (int i = 0; i < num_words; ++i) {
    pq.emplace(word_count[i], i, nullptr, nullptr);
  }
  for (int i = 0; i < num_words - 1; ++i) {
    auto min1 = pq.top(); pq.pop();
    auto min2 = pq.top(); pq.pop();
    pq.emplace(min1.count + min2.count, i + num_words, &min1, &min2);
  }
  
  std::vector<std::tuple<PqItem, std::vector<bool>, std::vector<int>>> stack = {{pq.top(), {}, {}}};
  PqItem node;
  std::vector<bool> codes;
  std::vector<int> points;
  codes_.clear(); points_.clear();
  codes_.resize(num_words); points_.resize(num_words);
  int max_depth = 0;
  while (not stack.empty()) {
    std::tie(node, codes, points) = stack.back();
    stack.pop_back();
    int k = node.index;
    if (k < num_words) {
      codes_[k] = codes;
      points_[k] = points;
    } else {
      points.push_back(k - num_words);
      std::vector<bool> left_codes = codes;
      std::vector<bool> right_codes = codes;
      left_codes.push_back(false);
      right_codes.push_back(true);
      stack.push_back({node.left, left_codes, points});
      stack.push_back({node.right, right_codes, points});
    }
  }


}


}  // namespace cusim
