//===- DeviceApi.h - the GPU runtime names both runtimes share ------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The sources the CUDA and the HIP build of the Poseidon runtimes, calibrators
// and tests share name what the two runtimes name differently by one neutral
// name: gpu<Name> for the runtime and device code, gpublas<Name> and
// GPUBLAS_<NAME> for BLAS, GPU_<NAME> for BLAS element types. Each is the CUDA
// or the HIP name, given side by side, as the build is a CUDA or a HIP build
// (__HIP__). HIP and hipBLAS keep CUDA's calling conventions, so the two names
// of a row take the same arguments. Code for one runtime only uses that
// runtime's own names. POSEIDON_GPU_CHECK checks the status a call of either
// runtime returns.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_RUNTIME_DEVICE_API_H
#define POSEIDON_RUNTIME_DEVICE_API_H

#ifdef __HIP__
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#define POSEIDON_CUDA_OR_HIP(Cuda, Hip) Hip
#else
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define POSEIDON_CUDA_OR_HIP(Cuda, Hip) Cuda
#endif

// clang-format off
// Devices, memory, streams, events, launches and errors.
#define gpuDevAttrClockRate POSEIDON_CUDA_OR_HIP(cudaDevAttrClockRate, hipDeviceAttributeClockRate)
#define gpuDeviceGetAttribute POSEIDON_CUDA_OR_HIP(cudaDeviceGetAttribute, hipDeviceGetAttribute)
#define gpuDeviceGetLimit POSEIDON_CUDA_OR_HIP(cudaDeviceGetLimit, hipDeviceGetLimit)
#define gpuDeviceProp_t POSEIDON_CUDA_OR_HIP(cudaDeviceProp, hipDeviceProp_t)
#define gpuDeviceSetLimit POSEIDON_CUDA_OR_HIP(cudaDeviceSetLimit, hipDeviceSetLimit)
#define gpuDeviceSynchronize POSEIDON_CUDA_OR_HIP(cudaDeviceSynchronize, hipDeviceSynchronize)
#define gpuErrorMemoryAllocation POSEIDON_CUDA_OR_HIP(cudaErrorMemoryAllocation, hipErrorOutOfMemory)
#define gpuError_t POSEIDON_CUDA_OR_HIP(cudaError_t, hipError_t)
#define gpuEventCreate POSEIDON_CUDA_OR_HIP(cudaEventCreate, hipEventCreate)
#define gpuEventCreateWithFlags POSEIDON_CUDA_OR_HIP(cudaEventCreateWithFlags, hipEventCreateWithFlags)
#define gpuEventDestroy POSEIDON_CUDA_OR_HIP(cudaEventDestroy, hipEventDestroy)
#define gpuEventDisableTiming POSEIDON_CUDA_OR_HIP(cudaEventDisableTiming, hipEventDisableTiming)
#define gpuEventElapsedTime POSEIDON_CUDA_OR_HIP(cudaEventElapsedTime, hipEventElapsedTime)
#define gpuEventRecord POSEIDON_CUDA_OR_HIP(cudaEventRecord, hipEventRecord)
#define gpuEventSynchronize POSEIDON_CUDA_OR_HIP(cudaEventSynchronize, hipEventSynchronize)
#define gpuEvent_t POSEIDON_CUDA_OR_HIP(cudaEvent_t, hipEvent_t)
#define gpuFree POSEIDON_CUDA_OR_HIP(cudaFree, hipFree)
#define gpuGetDevice POSEIDON_CUDA_OR_HIP(cudaGetDevice, hipGetDevice)
#define gpuGetDeviceCount POSEIDON_CUDA_OR_HIP(cudaGetDeviceCount, hipGetDeviceCount)
#define gpuGetDeviceProperties POSEIDON_CUDA_OR_HIP(cudaGetDeviceProperties, hipGetDeviceProperties)
#define gpuGetErrorString POSEIDON_CUDA_OR_HIP(cudaGetErrorString, hipGetErrorString)
#define gpuGetLastError POSEIDON_CUDA_OR_HIP(cudaGetLastError, hipGetLastError)
#define gpuLaunchKernel POSEIDON_CUDA_OR_HIP(cudaLaunchKernel, hipLaunchKernel)
#define gpuLimitMallocHeapSize POSEIDON_CUDA_OR_HIP(cudaLimitMallocHeapSize, hipLimitMallocHeapSize)
#define gpuMalloc POSEIDON_CUDA_OR_HIP(cudaMalloc, hipMalloc)
#define gpuMemGetInfo POSEIDON_CUDA_OR_HIP(cudaMemGetInfo, hipMemGetInfo)
#define gpuMemcpy POSEIDON_CUDA_OR_HIP(cudaMemcpy, hipMemcpy)
#define gpuMemcpyDeviceToHost POSEIDON_CUDA_OR_HIP(cudaMemcpyDeviceToHost, hipMemcpyDeviceToHost)
#define gpuMemcpyHostToDevice POSEIDON_CUDA_OR_HIP(cudaMemcpyHostToDevice, hipMemcpyHostToDevice)
#define gpuMemcpyToSymbol POSEIDON_CUDA_OR_HIP(cudaMemcpyToSymbol, hipMemcpyToSymbol)
#define gpuMemset POSEIDON_CUDA_OR_HIP(cudaMemset, hipMemset)
#define gpuMemsetAsync POSEIDON_CUDA_OR_HIP(cudaMemsetAsync, hipMemsetAsync)
#define gpuStreamCreateWithFlags POSEIDON_CUDA_OR_HIP(cudaStreamCreateWithFlags, hipStreamCreateWithFlags)
#define gpuStreamDestroy POSEIDON_CUDA_OR_HIP(cudaStreamDestroy, hipStreamDestroy)
#define gpuStreamNonBlocking POSEIDON_CUDA_OR_HIP(cudaStreamNonBlocking, hipStreamNonBlocking)
#define gpuStreamSynchronize POSEIDON_CUDA_OR_HIP(cudaStreamSynchronize, hipStreamSynchronize)
#define gpuStreamWaitEvent POSEIDON_CUDA_OR_HIP(cudaStreamWaitEvent, hipStreamWaitEvent)
#define gpuStream_t POSEIDON_CUDA_OR_HIP(cudaStream_t, hipStream_t)
#define gpuSuccess POSEIDON_CUDA_OR_HIP(cudaSuccess, hipSuccess)

// BLAS.
#define GPUBLAS_COMPUTE_32F POSEIDON_CUDA_OR_HIP(CUBLAS_COMPUTE_32F, HIPBLAS_COMPUTE_32F)
#define GPUBLAS_COMPUTE_32I POSEIDON_CUDA_OR_HIP(CUBLAS_COMPUTE_32I, HIPBLAS_COMPUTE_32I)
#define GPUBLAS_GEMM_DEFAULT POSEIDON_CUDA_OR_HIP(CUBLAS_GEMM_DEFAULT, HIPBLAS_GEMM_DEFAULT)
#define GPUBLAS_OP_N POSEIDON_CUDA_OR_HIP(CUBLAS_OP_N, HIPBLAS_OP_N)
#define GPUBLAS_OP_T POSEIDON_CUDA_OR_HIP(CUBLAS_OP_T, HIPBLAS_OP_T)
#define GPUBLAS_STATUS_SUCCESS POSEIDON_CUDA_OR_HIP(CUBLAS_STATUS_SUCCESS, HIPBLAS_STATUS_SUCCESS)
#define gpublasComputeType_t POSEIDON_CUDA_OR_HIP(cublasComputeType_t, hipblasComputeType_t)
#define gpublasCreate POSEIDON_CUDA_OR_HIP(cublasCreate, hipblasCreate)
#define gpublasDgemm POSEIDON_CUDA_OR_HIP(cublasDgemm, hipblasDgemm)
#define gpublasGemmEx POSEIDON_CUDA_OR_HIP(cublasGemmEx, hipblasGemmEx)
#define gpublasGemmStridedBatchedEx POSEIDON_CUDA_OR_HIP(cublasGemmStridedBatchedEx, hipblasGemmStridedBatchedEx)
#define gpublasGetStatusString POSEIDON_CUDA_OR_HIP(cublasGetStatusString, hipblasStatusToString)
#define gpublasHandle_t POSEIDON_CUDA_OR_HIP(cublasHandle_t, hipblasHandle_t)
#define gpublasOperation_t POSEIDON_CUDA_OR_HIP(cublasOperation_t, hipblasOperation_t)
#define gpublasSetStream POSEIDON_CUDA_OR_HIP(cublasSetStream, hipblasSetStream)
#define gpublasStatus_t POSEIDON_CUDA_OR_HIP(cublasStatus_t, hipblasStatus_t)

// Element types of BLAS operands.
#define GPU_R_16BF POSEIDON_CUDA_OR_HIP(CUDA_R_16BF, HIP_R_16BF)
#define GPU_R_16F POSEIDON_CUDA_OR_HIP(CUDA_R_16F, HIP_R_16F)
#define GPU_R_32F POSEIDON_CUDA_OR_HIP(CUDA_R_32F, HIP_R_32F)
#define GPU_R_32I POSEIDON_CUDA_OR_HIP(CUDA_R_32I, HIP_R_32I)
#define GPU_R_8I POSEIDON_CUDA_OR_HIP(CUDA_R_8I, HIP_R_8I)
#define gpuDataType_t POSEIDON_CUDA_OR_HIP(cudaDataType_t, hipDataType)
// clang-format on

// A runtime call that does not return gpuSuccess stops the program through
// poseidon::fatalCall, which a source using this includes from
// profile/ProfileFormat.h. An ignored failure leaves a sticky error that the
// next caller of gpuGetLastError reports as its own.
#define POSEIDON_GPU_CHECK(call)                                               \
  do {                                                                         \
    gpuError_t e_ = (call);                                                    \
    if (e_ != gpuSuccess)                                                      \
      poseidon::fatalCall(#call, __FILE__, __LINE__, gpuGetErrorString(e_));   \
  } while (0)

// ROCm's hip_fp16.h and hip_bf16.h call device-library functions this clang
// does not declare. A HIP build needs the two formats only as storage and as
// the target of one conversion from double, which the language's own types
// perform with a single rounding.
// clang-format off
#define gpuHalf POSEIDON_CUDA_OR_HIP(__half, _Float16)
#define gpuBfloat16 POSEIDON_CUDA_OR_HIP(__nv_bfloat16, __bf16)
#define gpuDoubleToHalf(v) POSEIDON_CUDA_OR_HIP(__double2half(v), static_cast<_Float16>(v))
#define gpuDoubleToBfloat16(v) POSEIDON_CUDA_OR_HIP(__double2bfloat16(v), static_cast<__bf16>(v))
// clang-format on

// Device code.
// clang-format off
#define gpuTrap POSEIDON_CUDA_OR_HIP(__trap, __builtin_trap)
// clang-format on
// A shuffle down within 32 lanes: a CUDA warp, and the 32-lane half of a
// 64-lane wavefront, where a source lane past the half returns the caller's
// own value as on a warp. The mask is CUDA's; HIP has none.
// clang-format off
#define gpuShflDownSync(mask, var, delta) POSEIDON_CUDA_OR_HIP(__shfl_down_sync(mask, var, delta), __shfl_down(var, delta, 32))
// clang-format on

#endif // POSEIDON_RUNTIME_DEVICE_API_H
