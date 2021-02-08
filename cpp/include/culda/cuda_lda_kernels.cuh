// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include "utils/cuda_utils_kernels.cuh"

namespace cusim {

__inline__ __device__
cuda_scalar Psi(cuda_scalar x) {
  
}

__global__ void EstepKernel(
  const int* indices, const int* indptr, 
  const int num_indices, const int num_indptr,
  const int num_words, const int num_topics, const int num_iters,
  cuda_scalar* gamma, cuda_scalar* new_gamma, cuda_scalar* phi,
  cuda_scalar* alpha, cuda_scalar* beta,
  cuda_scalar* grad_alpha, cuda_scalar* new_beta) {
  
  // storage for block
  cuda_scalar* _gamma = gamma + num_topics * blockIdx.x;
  cuda_scalar* _new_gamma = new_gamma + num_topics * blockIdx.x;
  cuda_scalar* _phi = phi + num_topics * blockIdx.x;

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
          _phi[l] = beta[w * num_topics + l] * exp(Psi(_gamma[l]));
        __syncthreads();
        
        // normalize phi and add it to new gamma and new beta
        cuda_scalar phi_sum = Sum(_phi, num_topics);
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
    cuda_scalar gamma_sum = Sum(_gamma, num_topics);
    for (int j = threadIdx.x; j < num_topics, j += blockDim.x)
      grad_alpha[j] += (Psi(_gamma[j]) - Psi(gamma_sum));
    __syncthreaads()
  } 
}

}  // cusim
