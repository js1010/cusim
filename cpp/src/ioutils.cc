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

bool IoUtils::Init(std::string opt_path) {
  std::ifstream in(opt_path.c_str());
  if (not in.is_open()) return false;

  std::string str((std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>());
  std::string err_cmt;
  auto _opt = json11::Json::parse(str, err_cmt);
  if (not err_cmt.empty()) return false;
  opt_ = _opt;
  CuSimLogger().set_log_level(opt_["c_log_level"].int_value());
  return true;
}

void IoUtils::ParseLine(std::string line, std::vector<std::string>& ret) {
  ret.clear();
  int n = line.size();
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
}

int IoUtils::LoadStreamFile(std::string filepath) {
  INFO("read gensim file to generate vocabulary: {}", filepath);
  stream_fin_.open(filepath.c_str());
  int count = 0;
  std::string line;
  while (getline(stream_fin_, line))
    count++;
  stream_fin_.close();
  stream_fin_.open(filepath.c_str());
  word_idmap_.clear();
  word_list_.clear();
  word_count_.clear();
  return count;
}

int IoUtils::ReadStreamForVocab(int num_lines) {
  int read_cnt = 0;
  std::string line;
  std::vector<std::string> line_vec;
  while (getline(stream_fin_, line) and read_cnt < num_lines) {
    ParseLine(line, line_vec);
    for (auto& word: line_vec) {
      if (not word_count_.count(word)) word_count_[word] = 0;
      word_count_[word]++;
    }
    read_cnt++;
  }
  if (read_cnt < num_lines) stream_fin_.close();
  return read_cnt;
}

void IoUtils::GetWordVocab(int min_count) {
  INFO("number of raw words: {}", word_count_.size());
  for (auto& it: word_count_) {
    if (it.second >= min_count) {
      word_idmap_[it.first] = word_idmap_.size();
      word_list_.push_back(it.first);
    }
  }
  INFO("number of words after filtering: {}", word_list_.size());
}

}  // namespace cusim
