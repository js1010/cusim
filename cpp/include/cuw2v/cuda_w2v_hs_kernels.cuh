// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "utils/cuda_utils_kernels.cuh"
#include "cuw2v/cuda_w2v_base_kernels.cuh"


namespace cusim {

__global__ void W2VHsSgKernel(
  const int* cols, const int* indptr,
  const bool* codes, const int* points, const int* hs_indptr,
  const int num_indptr, const int num_dims, const int window_size,
  default_random_engine* rngs,
  float* emb_in, float* emb_out, 
  float* loss_nume, float* loss_deno, const float lr) {
  
  default_random_engine& rng = rngs[blockIdx.x];
  float& _loss_nume = loss_nume[blockIdx.x];
  float& _loss_deno = loss_deno[blockIdx.x];

  uniform_int_distribution<int> dist_window(0, window_size - 1);
  static __shared__ int reduced_windows;
  extern __shared__ float shared_memory[];
  float* grad = &shared_memory[0];

  // zero-initialize shared mem
  for (int i = threadIdx.x; i < num_dims; i += blockDim.x)
    grad[i] = 0.0f;
  __syncthreads();

  for (int i = blockIdx.x; i < num_indptr; i += gridDim.x) {
    int beg = indptr[i], end = indptr[i + 1];
    for (int j = beg; j < end; ++j) {
      if (threadIdx.x == 0) reduced_windows = dist_window(rng);
      __syncthreads();
      int beg2 = max(beg, j - window_size + reduced_windows);
      int end2 = min(end, j + window_size - reduced_windows + 1);
      for (int k = beg2; k < end2; ++k) {
        if (k == j) continue;
        float* _emb_in = emb_in + num_dims * cols[k];
        int beg3 = hs_indptr[cols[j]];
        int end3 = hs_indptr[cols[j] + 1];
        for (int l = beg3; l < end3; ++l) {
          if (codes[l]) {
            PositiveFeedback(_emb_in, emb_out + num_dims * points[l],
                grad, _loss_nume, _loss_deno, num_dims, lr);
          } else {
            NegativeFeedback(_emb_in, emb_out + num_dims * points[l],
                grad, _loss_nume, _loss_deno, num_dims, lr);
          }
          __syncthreads();
        }
        for (int l = threadIdx.x; l < num_dims; l += blockDim.x) {
          _emb_in[l] += grad[l];
          grad[l] = 0.0f;
        }
        __syncthreads();
      }
    }
  } 
}

__global__ void W2VHsCbowKernel(
  const int* cols, const int* indptr,
  const bool* codes, const int* points, const int* hs_indptr,
  const int num_indptr, const int num_dims, const int window_size, default_random_engine* rngs,
  float* emb_in, float* emb_out, 
  float* loss_nume, float* loss_deno, 
  const bool cbow_mean, const float lr) {
  
  default_random_engine& rng = rngs[blockIdx.x];
  float& _loss_nume = loss_nume[blockIdx.x];
  float& _loss_deno = loss_deno[blockIdx.x];

  uniform_int_distribution<int> dist_window(0, window_size - 1);
  static __shared__ int reduced_windows;
  extern __shared__ float shared_memory[];
  float* grad = &shared_memory[0];
  float* cbow = &shared_memory[num_dims];

  __syncthreads();

  for (int i = blockIdx.x; i < num_indptr; i += gridDim.x) {
    int beg = indptr[i], end = indptr[i + 1];
    for (int j = beg; j < end; ++j) {
      if (threadIdx.x == 0) reduced_windows = dist_window(rng);
      __syncthreads();
      int beg2 = max(beg, j - window_size + reduced_windows);
      int end2 = min(end, j + window_size - reduced_windows + 1);
      if (end2 - beg2 <= 1) continue;
      
      // zero-initialize shared mem
      for (int k = threadIdx.x; k < num_dims; k += blockDim.x) {
        grad[k] = 0.0f;
        cbow[k] = 0.0f;
      }
      
      // compute cbow
      for (int k = beg2; k < end2; ++k) {
        if (k == j) continue;
        for (int l = threadIdx.x; l < num_dims; l += blockDim.x) {
          cbow[l] += emb_in[num_dims * cols[k] + l];
        }
      }
      if (cbow_mean) {
        for (int k = threadIdx.x; k < num_dims; k += blockDim.x) {
          cbow[k] /= (end2 - beg2 - 1);
        }
      }
      __syncthreads();
       
      int beg3 = hs_indptr[cols[j]];
      int end3 = hs_indptr[cols[j] + 1];
      for (int k = beg3; k < end3; ++k) {
        if (codes[k]) {
          PositiveFeedback(cbow, emb_out + num_dims * points[k],
              grad, _loss_nume, _loss_deno, num_dims, lr);
        } else {
          NegativeFeedback(cbow, emb_out + num_dims * points[k],
              grad, _loss_nume, _loss_deno, num_dims, lr);
        }
        __syncthreads();
      }
      
      // normalize grad if cbow_mean = true
      if (cbow_mean) {
        for (int k = threadIdx.x; k < num_dims; k += blockDim.x) {
          grad[k] /= (end2 - beg2 - 1);
        }
      }
      __syncthreads();

      // update emb_in
      for (int k = beg2; k < end2; ++k) {
        if (k == j) continue;
        for (int l = threadIdx.x; l < num_dims; l += blockDim.x) {
          emb_in[num_dims * cols[k] + l] += grad[l];
        } 
        __syncthreads();
      }
    }
  } 
}

}  // cusim
