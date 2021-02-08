// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "utils/cuda_utils_kernels.cuh"


namespace cusim {

__inline__ __device__
float Digamma(float x) {
  float result = 0f, xx, xx2, xx4;
  for ( ; x < 7.0f; ++x)
    result -= 1.0f / x;
  x -= 0.5f;
  xx = 1.0f / x;
  xx2 = xx * xx;
  xx4 = xx2 * xx2;
  result += logf(x) + 1.0f / 24.0f * xx2 - 7.0f / 960.0f * xx4 + 
    31.0f / 8064.0f * xx4 * xx2 - 127.0f / 30720.0f * xx4 * xx4;
  return result;
}

__global__ void EstepKernel(
  const int* indices, const int* indptr, 
  const int num_indices, const int num_indptr,
  const int num_words, const int num_topics, const int num_iters,
  float* gamma, float* new_gamma, float* phi,
  float* alpha, float* beta,
  float* grad_alpha, float* new_beta) {
  
  // storage for block
  float* _gamma = gamma + num_topics * blockIdx.x;
  float* _new_gamma = new_gamma + num_topics * blockIdx.x;
  float* _phi = phi + num_topics * blockIdx.x;

  for (int i = blockIdx.x; i < num_indptr; i += gridDim.x) {
    int beg = indptr[i], end = indptr[i + 1];
    
    // initialize gamma
    for (int j = threadIdx.x; j < num_topics; j += blockDim.x)
      _gamma[j] = alpha[j] + (end - beg) / num_topics;
    __syncthreads();

    // iterate E step
    for (int j = 0; j < num_iters; ++j) {
      // initialize new gamma
      for (int k = threadIdx.x; k < num_topics; k += blockDim.x)
        _new_gamma[k] = 0;
      __synctheads();

      // compute phi from gamma
      for (int k = beg; k < end; ++k) {
        int w = indices[k];
        // compute phi
        for (int l = threadIdx.x; l < num_topics; l += blockDim.x)
          _phi[l] = beta[w * num_topics + l] * expf(Digamma(_gamma[l]));
        __syncthreads();
        
        // normalize phi and add it to new gamma and new beta
        float phi_sum = ReduceSum(_phi, num_topics);
        for (int l = threadIdx.x; l < num_topics; l += blockDim.x) {
          _phi[l] /= phi_sum;
          _new_gamma[l] += _phi[l];
          if (j + 1 == num_iters) new_beta[w * num_topics + l] += phi[l];
        }
        __syncthreads();
      }

      // update gamma
      for (int k = threadIdx.x; k < num_topics; l += blockDim.x)
        _gamma[k] = _new_gamma[k] + alpha[k];
      __syncthreads();
    }
    float gamma_sum = ReduceSum(_gamma, num_topics);
    for (int j = threadIdx.x; j < num_topics, j += blockDim.x)
      grad_alpha[j] += (Psi(_gamma[j]) - Psi(gamma_sum));
    __syncthreaads()
  } 
}

__global__ void MstepKernel(
  float* alpha, float* beta,
  float* grad_alpha, float* new_beta,
  const int num_words, const int num_topic) {
  

}

}  // cusim
