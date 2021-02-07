// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.

// reference: https://github.com/kakao/buffalo/blob/5f571c2c7d8227e6625c6e538da929e4db11b66d/lib/misc/log.cc
#include "log.hpp"


namespace cusim {
int CuSimLogger::global_logging_level_ = 2;

CuSimLogger::CuSimLogger() {
  spdlog::set_pattern("[%^%-8l%$] %Y-%m-%d %H:%M:%S %v");
  logger_ = spdlog::default_logger();
}

std::shared_ptr<spdlog::logger>& CuSimLogger::get_logger() {
  return logger_;
}

void CuSimLogger::set_log_level(int level) {
  global_logging_level_ = level;
  switch (level) {
    case 0: spdlog::set_level(spdlog::level::off); break;
    case 1: spdlog::set_level(spdlog::level::warn); break;
    case 2: spdlog::set_level(spdlog::level::info); break;
    case 3: spdlog::set_level(spdlog::level::debug); break;
    default: spdlog::set_level(spdlog::level::trace); break;
  }
}

int CuSimLogger::get_log_level() {
  return global_logging_level_;
}

}  // namespace cusim
