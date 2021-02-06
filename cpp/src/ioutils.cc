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
  num_lines_ = count;
  remain_lines_ = num_lines_;
  return count;
}

std::pair<int, int> IoUtils::ReadStreamForVocab(int num_lines, int num_threads) {
  int read_lines = std::min(num_lines, remain_lines_);
  remain_lines_ -= read_lines;
  #pragma omp parallel num_threads(num_threads)
  {
    std::string line;
    std::vector<std::string> line_vec;
    std::unordered_map<std::string, int> word_count;
    #pragma omp for schedule(dynamic, 4)
    for (int i = 0; i < read_lines; ++i) {
      // get line thread-safely
      {
        std::unique_lock<std::mutex> lock(global_lock_);
        getline(stream_fin_, line);
      }

      // seems to bottle-neck
      ParseLine(line, line_vec);

      // update private word count
      for (auto& word: line_vec) {
        word_count[word]++;
      }
    }

    // update word count to class variable
    {
      std::unique_lock<std::mutex> lock(global_lock_);
      for (auto& it: word_count) {
        word_count_[it.first] += it.second;
      }
    }
  }
  return {read_lines, remain_lines_};
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
