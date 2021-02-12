// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "utils/cuda_utils_kernels.cuh"
#include "w2v/cuda_w2v_base_kernels.cuh"

using thrust::random::default_random_engine;
using thrust::random::uniform_int_distribution;

namespace cusim {

__global__ void W2VNegSgKernel(
  const int* cols, const int* indptr, const int window,
  const int* random_table, const int random_size, default_random_engine* rngs,
  const int num_cols, const int num_indptr, const int num_dims, const int neg,
  float* emb_in, float* emb_out, float* loss_nume, float* loss_deno) {
  
  default_random_engine& rng = rngs[blockIdx.x];
  float& _loss_nume = loss_nume[blockIdx.x];
  float& _loss_deno = loss_deno[blockIdx.x];

  static __shared__ uniform_int_distribution<int> dist_neg(0, random_size - 1);
  static __shared__ uniform_int_distribution<int> dist_window(0, window - 1);
  static __shared__ int reduced_windows;
  static __shared__ int neg_word;
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
      int beg2 = max(beg, j - window + reduced_windows);
      int end2 = min(end, j + window - reduced_windows + 1);
      float* _emb_in = emb_in + num_dims * cols[j];
      for (int k = beg2; k < end2; ++k) {
        if (k == j) continue;
        PositiveFeedback(_emb_in, emb_out + num_dims * cols[k], 
            grad, _loss_nume, _loss_deno, num_dims)
        if (int l = 0; l < neg; ++l) {
          if (threadIdx.x == 0) neg_word = random_table[dist_neg(rng)];
          __syncthreads();
          NegativeFeedback(_emb_in, emb_out + num_dims * neg_word, 
              grad, _loss_nume, _loss_deno, num_dims);
        }
        __syncthreads();
        for (int l = threadIdx.x; l < num_dims; l += blockDim.x) {
          emb_in[num_dims * j + l] += grad[l];
          grad[l] = 0.0f;
        }
        __syncthreads();
      }
    }
  } 
}

__global__ void W2VNegCbowKernel(
  const int* cols, const int* indptr, const int window,
  const int* random_table, const int random_size, default_random_engine* rngs,
  const int num_cols, const int num_indptr, const int num_dims, const int neg,
  float* emb_in, float* emb_out, float* loss_nume, float* loss_deno, const bool use_mean) {
  
  default_random_engine& rng = rngs[blockIdx.x];
  float& _loss_nume = loss_nume[blockIdx.x];
  float& _loss_deno = loss_deno[blockIdx.x];

  static __shared__ uniform_int_distribution<int> dist_neg(0, random_size - 1);
  static __shared__ uniform_int_distribution<int> dist_window(0, window - 1);
  static __shared__ int reduced_windows;
  static __shared__ int neg_word;
  extern __shared__ float shared_memory[];
  float* grad = &shared_memory[0];
  float* cbow = &shared_memory[num_dims];

  __syncthreads();

  for (int i = blockIdx.x; i < num_indptr; i += gridDim.x) {
    int beg = indptr[i], end = indptr[i + 1];
    for (int j = beg; j < end; ++j) {
      if (threadIdx.x == 0) reduced_windows = dist_window(rng);
      __syncthreads();
      int beg2 = max(beg, j - window + reduced_windows);
      int end2 = min(end, j + window - reduced_windows + 1);
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
      if (use_mean) {
        for (int k = threadIdx.x; k < num_dims; k += blockDim.x) {
          cbow[k] /= (end2 - beg2 - 1);
        }
      }
      __syncthreads();
      
      PositiveFeedback(cbow, emb_out + num_dims * cols[j], grad,
          loss_nume, loss_deno, num_dims);
      __syncthreads();
      
      // update negative feedback
      for (int k = 0; k < neg; ++k){
        if (threadIdx.x == 0) neg_word = random_table[dist_neg(rng)];
        __syncthredas();
        NegativeFeedback(cbow, emb_out + num_dims * neg_word, 
            grad, _loss_nume, _loss_deno, num_dims);
      }
      __syncthreads();
      
      // normalize grad if use_mean = true
      if (use_mean) {
        for (int k = threadIdx.x; k < num_dims; k += blockDim.x) {
          grad[k] /= (end2 - beg2 - 1);
        }
      }
      __syncthreads();

      // update emb_in
      for (int k = beg2; k < end2; ++k) {
        if (k == j) continue; 
        for (int l = threadIdx.x; l < num_dims; l += blockDim.x)
          emb_in[num_dims * cols[k] + l] += grad[l];
      }
      __syncthreads();

    }
  } 
}

}  // cusim
