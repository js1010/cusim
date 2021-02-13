// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.

// reference: https://github.com/kakao/buffalo/blob/5f571c2c7d8227e6625c6e538da929e4db11b66d/lib/misc/log.cc
#include "utils/log.hpp"


namespace cusim {
int CuSimLogger::global_logging_level_ = 2;

CuSimLogger::CuSimLogger() {
  spdlog::set_pattern("[%^%-8l%$] %Y-%m-%d %H:%M:%S %v");
  logger_ = spdlog::default_logger();
}

CuSimLogger::CuSimLogger(std::string name) {
  // auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
  // spdlog::sinks_init_list sinks = {console_sink, stderr_sink};
  logger_ = std::make_shared<spdlog::logger>(name, stderr_sink);
  logger_->set_pattern("[%^%-8l%$] %Y-%m-%d %H:%M:%S %v");
}

std::shared_ptr<spdlog::logger>& CuSimLogger::get_logger() {
  return logger_;
}

void CuSimLogger::set_log_level(int level) {
  global_logging_level_ = level;
  switch (level) {
    case 0: logger_->set_level(spdlog::level::off); break;
    case 1: logger_->set_level(spdlog::level::warn); break;
    case 2: logger_->set_level(spdlog::level::info); break;
    case 3: logger_->set_level(spdlog::level::debug); break;
    default: logger_->set_level(spdlog::level::trace); break;
  }
}

int CuSimLogger::get_log_level() {
  return global_logging_level_;
}

}  // namespace cusim
