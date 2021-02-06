// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "ioutils.hpp"

namespace cusim {

IoUtils::IoUtils() {
  logger_ = CuSimLogger().get_logger();
}

IoUtils::~IoUtils() {}

std::vector<std::string> IoUtils::parse_line(std::string line) {
  int n = line.size();
  std::vector<std::string> ret;
  std::string element;
  for (int i = 0; i < n; ++i) {
    if (line[i] == ' ') {
      ret.push_back(element);
      element.clear();
    } else {
      element += line[i];
    }
  }
  if (element.size() > 0) {
    ret.push_back(element);
  }
  return ret;
}

void IoUtils::LoadGensimVocab(std::string filepath, int min_count) {
  INFO("read gensim file to generate vocabulary: {}, min_count: {}", filepath, min_count);
  std::ifstream fin(filepath.c_str());
  std::unordered_map<std::string, int> word_count;
  while (not fin.eof()) {
    std::string line;
    getline(fin, line);
    std::vector<std::string> line_vec = parse_line(line);
    for (auto& word: line_vec) {
      if (not word_count.count(word)) word_count[word] = 0;
      word_count[word]++;
    }
  }
  INFO("number of raw words: {}", word_count.size());
  word_idmap_.clear();
  word_list_.clear();
  for (auto& it: word_count) {
    if (it.second >= min_count) {
      word_idmap_[it.first] = vocab_.size();
      word_list_.push_back(it.first);
    }
  }
  INFO("number of words after filtering: {}", word_list_.size());
}

}  // namespace cusim
