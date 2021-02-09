// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

struct DeviceInfo {
  int devId, mp_cnt, major, minor, cores;
  bool unknown = false;
};

#define WARP_SIZE 32
