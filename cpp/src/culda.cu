// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include "culda.cuh"

namespace cusim {

CuLDA::CuLDA() {
  logger_ = CuSimLogger().get_logger();
}

CuLDA::~CuLDA() {}

} // namespace cusim
