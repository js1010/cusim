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

namespace cusim {

class IoUtils {
 public:
  IoUtils();
  ~IoUtils();
  void LoadGensimVocab(std::string filepath, int min_count);
 private:
  std::vector<std::string> parse_line(std::string line);
  std::unordered_map<std::string, int> word_idmap_;
  std::vector<std::string> word_list_;
};  // class IoUtils

} // namespace cusim
