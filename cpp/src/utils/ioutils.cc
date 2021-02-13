// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "utils/ioutils.hpp"

namespace cusim {

IoUtils::IoUtils() {
  logger_container_.reset(new CuSimLogger("ioutils"));
  logger_ = logger_container_->get_logger();
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
  logger_container_->set_log_level(opt_["c_log_level"].int_value());
  lower_ = opt_["lower"].bool_value();
  return true;
}

void IoUtils::ParseLine(std::string line, std::vector<std::string>& ret) {
  ParseLineImpl(line, ret);
}


void IoUtils::ParseLineImpl(std::string line, std::vector<std::string>& ret) {
  ret.clear();
  int n = line.size();
  std::string element;
  for (int i = 0; i < n; ++i) {
    if (line[i] == ' ') {
      ret.push_back(element);
      element.clear();
    } else {
      element += (lower_? std::tolower(line[i]): line[i]);
    }
  }
  if (element.size() > 0) {
    ret.push_back(element);
  }
}

int64_t IoUtils::LoadStreamFile(std::string filepath) {
  INFO("read gensim file to generate vocabulary: {}", filepath);
  if (stream_fin_.is_open()) stream_fin_.close();
  stream_fin_.open(filepath.c_str());
  int64_t count = 0;
  std::string line;
  while (getline(stream_fin_, line))
    count++;
  stream_fin_.close();
  stream_fin_.open(filepath.c_str());
  num_lines_ = count;
  remain_lines_ = num_lines_;
  INFO("number of lines: {}", num_lines_);
  return count;
}

std::pair<int, int> IoUtils::TokenizeStream(int num_lines, int num_threads) {
  int read_lines = static_cast<int>(std::min(static_cast<int64_t>(num_lines), remain_lines_));
  if (not read_lines) return {0, 0};
  remain_lines_ -= read_lines;
  cols_.clear();
  cols_.resize(read_lines);
  indptr_.resize(read_lines);
  std::fill(indptr_.begin(), indptr_.end(), 0);
  #pragma omp parallel num_threads(num_threads)
  {
    std::string line;
    std::vector<std::string> line_vec;
    #pragma omp for schedule(dynamic, 4)
    for (int i = 0; i < read_lines; ++i) {
      // get line thread-safely
      {
        std::unique_lock<std::mutex> lock(global_lock_);
        getline(stream_fin_, line);
      }

      // seems to be bottle-neck
      ParseLine(line, line_vec);

      // tokenize
      for (auto& word: line_vec) {
        if (not word_idmap_.count(word)) continue;
        cols_[i].push_back(word_idmap_[word]);
      }
    }
  }
  int cumsum = 0;
  for (int i = 0; i < read_lines; ++i) {
    cumsum += cols_[i].size();
    indptr_[i] = cumsum;
  }
  return {read_lines, indptr_[read_lines - 1]};
}

void IoUtils::GetToken(int* rows, int* cols, int* indptr) {
  int n = cols_.size();
  for (int i = 0; i < n; ++i) {
    int beg = i == 0? 0: indptr_[i - 1];
    int end = indptr_[i];
    for (int j = beg; j < end; ++j) {
      rows[j] = i;
      cols[j] = cols_[i][j - beg];
    }
    indptr[i] = indptr_[i];
  }
}

std::pair<int, int> IoUtils::ReadStreamForVocab(int num_lines, int num_threads) {
  int read_lines = static_cast<int>(std::min(static_cast<int64_t>(num_lines), remain_lines_));
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

      // seems to be bottle-neck
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
  if (not remain_lines_) stream_fin_.close();
  return {read_lines, word_count_.size()};
}

void IoUtils::GetWordVocab(int min_count, std::string keys_path, std::string count_path) {
  INFO("number of raw words: {}", word_count_.size());
  word_idmap_.clear(); word_list_.clear();
  for (auto& it: word_count_) {
    if (it.second >= min_count) {
      word_idmap_[it.first] = word_idmap_.size();
      word_list_.push_back(it.first);
    }
  }
  INFO("number of words after filtering: {}", word_list_.size());

  // write keys and count to csv file
  std::ofstream fout1(keys_path.c_str());
  std::ofstream fout2(count_path.c_str());
  INFO("dump keys to {}", keys_path);
  int n = word_list_.size();
  for (int i = 0; i < n; ++i) {
    std::string line = word_list_[i] + "\n";
    fout1.write(line.c_str(), line.size());
    line = std::to_string(word_count_[word_list_[i]]) + "\n";
    fout2.write(line.c_str(), line.size());
  }
  fout1.close(); fout2.close();
}

}  // namespace cusim
