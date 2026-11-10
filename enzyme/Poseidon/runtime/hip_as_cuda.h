// The CUDA runtime and cuBLAS names the Poseidon runtimes are written in, for
// their HIP builds (POSEIDON_ROCM_PATH): HIP and hipBLAS keep CUDA's calling
// conventions, so each name maps to its HIP counterpart one for one. Only the
// names the runtimes, their tests and their calibrators use are mapped.
#ifndef POSEIDON_HIP_AS_CUDA_H
#define POSEIDON_HIP_AS_CUDA_H

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define cudaDeviceGetLimit hipDeviceGetLimit
#define cudaDeviceSetLimit hipDeviceSetLimit
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaGetErrorString hipGetErrorString
#define cudaLaunchKernel hipLaunchKernel
#define cudaLimitMallocHeapSize hipLimitMallocHeapSize
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

#define cublasComputeType_t hipblasComputeType_t
#define cublasCreate hipblasCreate
#define cublasDgemm hipblasDgemm
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define cublasSetStream hipblasSetStream
#define cublasStatus_t hipblasStatus_t
#define cudaDataType_t hipDataType
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32I HIPBLAS_COMPUTE_32I
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_32I HIP_R_32I
#define CUDA_R_8I HIP_R_8I

// ROCm's hip_fp16.h and hip_bf16.h call device-library functions this clang
// no longer declares. The runtimes need the two formats only as storage and as
// the target of one conversion from double, which the language's own types
// perform with a single rounding.
#define __half _Float16
#define __nv_bfloat16 __bf16
#define __double2half(v) static_cast<_Float16>(v)
#define __double2bfloat16(v) static_cast<__bf16>(v)

#endif // POSEIDON_HIP_AS_CUDA_H
