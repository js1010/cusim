// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "utils/cuda_utils_kernels.cuh"


namespace cusim {

// reference: http://web.science.mq.edu.au/~mjohnson/code/digamma.c
__inline__ __device__
float Digamma(float x) {
  float result = 0.0f, xx, xx2, xx4;
  for ( ; x < 7.0f; ++x)
    result -= 1.0f / x;
  x -= 0.5f;
  xx = 1.0f / x;
  xx2 = xx * xx;
  xx4 = xx2 * xx2;
  result += logf(x) + 1.0f / 24.0f * xx2 
    - 7.0f / 960.0f * xx4 + 31.0f / 8064.0f * xx4 * xx2 
    - 127.0f / 30720.0f * xx4 * xx4;
  return result;
}

__global__ void EstepKernel(
  const int* cols, const int* indptr, const bool* vali,
  const int num_cols, const int num_indptr,
  const int num_topics, const int num_iters,
  float* gamma, float* new_gamma, float* phi,
  const float* alpha, const float* beta,
  float* grad_alpha, float* new_beta, 
  float* train_losses, float* vali_losses, int* mutex) {
  
  // storage for block
  float* _gamma = gamma + num_topics * blockIdx.x;
  float* _new_gamma = new_gamma + num_topics * blockIdx.x;
  float* _phi = phi + num_topics * blockIdx.x;
  float* _grad_alpha = grad_alpha + num_topics * blockIdx.x;

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
        _new_gamma[k] = 0.0f;
      __syncthreads();

      // compute phi from gamma
      for (int k = beg; k < end; ++k) {
        const int w = cols[k];
        const bool _vali = vali[k];
        
        // compute phi
        if (not _vali or j + 1 == num_iters) {
          for (int l = threadIdx.x; l < num_topics; l += blockDim.x)
            _phi[l] = beta[w * num_topics + l] * expf(Digamma(_gamma[l]));
          __syncthreads();
          
          // normalize phi and add it to new gamma and new beta
          float phi_sum = ReduceSum(_phi, num_topics);

          for (int l = threadIdx.x; l < num_topics; l += blockDim.x) {
            _phi[l] /= phi_sum;
            if (not _vali) _new_gamma[l] += _phi[l];
          }
          __syncthreads();
        }
        
        if (j + 1 == num_iters) {
          // write access of w th vector of new_beta 
          if (threadIdx.x == 0) {
            while (atomicCAS(&mutex[w], 0, 1)) {}
          } 

          __syncthreads();
          for (int l = threadIdx.x; l < num_topics; l += blockDim.x) {
            if (j + 1 == num_iters) { 
              if (not _vali) new_beta[w * num_topics + l] += _phi[l];
              _phi[l] *= beta[w * num_topics + l];
            }
          }
          __syncthreads();

          // release lock
          if (threadIdx.x == 0) mutex[w] = 0;
          __syncthreads();

          float p = fmaxf(EPS, ReduceSum(_phi, num_topics));
          if (threadIdx.x == 0) {
            if (_vali)
              vali_losses[blockIdx.x] += logf(p);
            else
              train_losses[blockIdx.x] += logf(p);
          } 
        }
        __syncthreads();
      }

      // update gamma
      for (int k = threadIdx.x; k < num_topics; k += blockDim.x)
        _gamma[k] = _new_gamma[k] + alpha[k];
      __syncthreads();
    }
    float gamma_sum = ReduceSum(_gamma, num_topics);
    for (int j = threadIdx.x; j < num_topics; j += blockDim.x)
      _grad_alpha[j] += (Digamma(_gamma[j]) - Digamma(gamma_sum));

    __syncthreads();
  } 
}

}  // cusim
