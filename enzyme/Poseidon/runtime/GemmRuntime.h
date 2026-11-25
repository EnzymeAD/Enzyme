//===- GemmRuntime.h - what the GEMM dispatch runtimes share --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The error policy, the GEMM library handle and the scratch buffers of the
// runtimes that define the entry points of GemmDispatch.h.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_GEMM_RUNTIME_H
#define POSEIDON_GEMM_RUNTIME_H

#include "GemmDispatch.h"

#ifndef __HIP__
#include <cublas_v2.h>
#endif

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// A runtime error leaves C unwritten or wrong, so it stops the program.
[[noreturn]] inline void poseidonGemmFatal(const char *call, const char *file,
                                           int line, const char *status) {
  fflush(stdout);
  fprintf(stderr, "Poseidon GEMM runtime: %.*s failed at %s:%d: %s\n",
          (int)strcspn(call, "("), call, file, line, status);
  abort();
}

#define POSEIDON_GEMM_CK(call)                                                 \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess)                                                     \
      poseidonGemmFatal(#call, __FILE__, __LINE__, cudaGetErrorString(e_));    \
  } while (0)

#define POSEIDON_GEMM_CB(call)                                                 \
  do {                                                                         \
    cublasStatus_t s_ = (call);                                                \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                           \
      poseidonGemmFatal(#call, __FILE__, __LINE__, cublasGetStatusString(s_)); \
  } while (0)

// One handle for every runtime of the process, created on first use; the
// stream is set on every call.
inline cublasHandle_t poseidonCublas() {
  static cublasHandle_t h = [] {
    cublasHandle_t created;
    POSEIDON_GEMM_CB(cublasCreate(&created));
    return created;
  }();
  return h;
}

// Grows the device buffer *p of *cap bytes to at least `bytes`.
inline void poseidonReserve(void **p, size_t *cap, size_t bytes) {
  if (*cap >= bytes)
    return;
  if (*p)
    POSEIDON_GEMM_CK(cudaFree(*p));
  POSEIDON_GEMM_CK(cudaMalloc(p, bytes));
  *cap = bytes;
}

#endif // POSEIDON_GEMM_RUNTIME_H
