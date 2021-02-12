// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "utils/cuda_utils_kernels.cuh"

using thrust::random::default_random_engine;
using thrust::random::uniform_int_distribution;

namespace cusim {


__inline__ __device__
void PositiveFeedback(const float* vec1, float* vec2, float* grad, 
    float& loss_nume, float& loss_deno, const int num_dims, const float lr) {
  static __shared__ float g;
  float dot = Dot(emb_in[num_dims * j], emb_out[num_dims * k], num_dims);
  if (threadIdx.x == 0) {
    float exp_dot = expf(-dot);
    g = exp_dot / (1 + exp_dot) * lr;
    loss_nume += logf(1 + exp_dot);
    loss_deno++;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < num_dims; i += blockDim.x) {
    float tmp = vec2[i];
    vec2[i] += vec1[i] * g;
    grad[i] += tmp * g;
  }
  __syncthreads();
}

__inline__ __device__
void NegativeFeedback(const float* vec1, float* vec2, float* grad, 
    float& loss_nume, float& loss_deno, const int num_dims, const float lr) {
  static __shared__ float g;
  float dot = Dot(emb_in[num_dims * j], emb_out[num_dims * k], num_dims);
  if (threadIdx.x == 0) {
    float exp_dot = expf(dot);
    g = exp_dot / (1 + exp_dot) * lr;
    loss_nume += logf(1 + exp_dot);
    loss_deno++;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < num_dims; i += blockDim.x) {
    float tmp = vec2[i];
    vec2[i] -= vec1[i] * g;
    grad[i] -= tmp * g;
  }
  __syncthreads();
}

}  // cusim
