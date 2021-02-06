// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <set>
#include <random>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <queue>
#include <deque>
#include <functional>
#include <vector>
#include <cmath>
#include <chrono> // NOLINT
#include <iostream>
#include <unordered_map>

#include "json11.hpp"
#include "log.hpp"
#include "types.hpp"

namespace cusim {

class IoUtils {
 public:
  IoUtils();
  ~IoUtils();
  bool Init(std::string opt_path);
  int LoadStreamFile(std::string filepath);
  int ReadStreamForVocab(int num_lines);
  void GetWordVocab(int min_count);
 private:
  void ParseLine(std::string line, std::vector<std::string>& line_vec);

  std::ifstream stream_fin_;
  json11::Json opt_;
  std::shared_ptr<spdlog::logger> logger_;
  std::unordered_map<std::string, int> word_idmap_, word_count_;
  std::vector<std::string> word_list_;
};  // class IoUtils

} // namespace cusim
