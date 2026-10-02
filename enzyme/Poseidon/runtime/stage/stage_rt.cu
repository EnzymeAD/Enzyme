//===- stage_rt.cu - df64 parameter-array staging runtime ----------------===//
//
// Companion to StageParam.cpp: the host cc1 prepends a call
// to __poseidon_stage_split_f64 to every launch of a kernel rewritten to read
// a pointer parameter as {hi@+0, lo@+4} FP32 limbs; the returned scratch
// buffer replaces the original pointer in that launch's arguments.
//
// The byte count is read from the CUDA allocation the pointer belongs to
// (cuMemGetAddressRange), so an interior pointer stages the tail of its own
// allocation; a pointer that is not a device allocation aborts, since the
// kernel about to launch reads limbs and there is no correct fallback. Slot k
// holds the Dekker split of double k, bit for bit the pair the in-kernel split
// produced, so hi + lo reconstructs v exactly.
//===---------------------------------------------------------------------===//

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace {

__global__ void poseidon_stage_split_f64_kernel(const double *__restrict__ src,
                                                float2 *__restrict__ dst,
                                                size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double v = src[i];
  float hi = (float)v;
  // Exact by Dekker: v - (double)hi is representable, and its round-to-nearest
  // FP32 image is the residual limb. No multiply-add appears here, so no
  // contraction setting can perturb it.
  float lo = (float)(v - (double)hi);
  dst[i] = make_float2(hi, lo);
}

struct StageBuf {
  void *ptr = nullptr;
  size_t bytes = 0;
};

std::mutex &stageMutex() {
  static std::mutex m;
  return m;
}

std::map<const void *, StageBuf> &stageBufs() {
  static std::map<const void *, StageBuf> m;
  return m;
}

[[noreturn]] void stageAbort(const char *what, const void *p) {
  std::fprintf(stderr,
               "[poseidon-stage] %s for device pointer %p. The launching "
               "kernel was rewritten to read df64 limbs, so there is no "
               "correct value to pass it; aborting rather than launching with "
               "wrong data.\n",
               what, p);
  std::abort();
}

// cuMemGetAddressRange through the runtime's driver-entry-point shim, so the
// object only needs -lcudart.
size_t allocationTailBytes(const void *src) {
  typedef CUresult(CUDAAPI * AddrRangeFn)(CUdeviceptr *, size_t *, CUdeviceptr);
  static AddrRangeFn fn = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    void *p = nullptr;
#if CUDART_VERSION >= 12000
    cudaDriverEntryPointQueryResult qr;
#if CUDART_VERSION >= 12050
    // Ask for the 12000 ABI explicitly: on a driver older than the toolkit
    // (driver 13.0 with toolkit 13.1) the unversioned lookup returns
    // cudaErrorInvalidValue and a null pointer while the versioned lookup
    // succeeds, and cudaGetDriverEntryPoint is deprecated from CUDA 13. The
    // unversioned call stays as the fallback for a CUDA 12 box.
    if (cudaGetDriverEntryPointByVersion("cuMemGetAddressRange", &p, 12000,
                                         cudaEnableDefault,
                                         &qr) != cudaSuccess ||
        qr != cudaDriverEntryPointSuccess)
      p = nullptr;
    if (!p)
#endif
    if (cudaGetDriverEntryPoint("cuMemGetAddressRange", &p, cudaEnableDefault,
                                &qr) != cudaSuccess ||
        qr != cudaDriverEntryPointSuccess)
      p = nullptr;
#else
    if (cudaGetDriverEntryPoint("cuMemGetAddressRange", &p,
                                cudaEnableDefault) != cudaSuccess)
      p = nullptr;
#endif
    fn = (AddrRangeFn)p;
  });
  if (!fn)
    stageAbort("the CUDA driver entry point cuMemGetAddressRange is unavailable",
               src);
  CUdeviceptr base = 0;
  size_t size = 0;
  if (fn(&base, &size, (CUdeviceptr)(uintptr_t)src) != CUDA_SUCCESS)
    stageAbort("cuMemGetAddressRange failed", src);
  uintptr_t off = (uintptr_t)src - (uintptr_t)base;
  if (off > size)
    stageAbort("the pointer lies outside the allocation it reports", src);
  return size - off;
}

} // namespace

extern "C" void *__poseidon_stage_split_f64(const void *src, void *stream) {
  if (!src)
    return nullptr;
  size_t bytes = allocationTailBytes(src);
  // Whole 8-byte slots only; a trailing partial slot holds no double.
  size_t n = bytes / sizeof(double);
  if (n == 0)
    stageAbort("the allocation holds no whole FP64 element", src);

  void *dst = nullptr;
  {
    std::lock_guard<std::mutex> g(stageMutex());
    StageBuf &b = stageBufs()[src];
    if (b.ptr && b.bytes < n * sizeof(double)) {
      cudaFree(b.ptr);
      b.ptr = nullptr;
      b.bytes = 0;
    }
    if (!b.ptr) {
      if (cudaMalloc(&b.ptr, n * sizeof(double)) != cudaSuccess)
        stageAbort("cudaMalloc of the df64 limb buffer failed", src);
      b.bytes = n * sizeof(double);
    }
    dst = b.ptr;
  }

  const int threads = 256;
  size_t blocks = (n + threads - 1) / threads;
  poseidon_stage_split_f64_kernel<<<(unsigned)blocks, threads, 0,
                                    (cudaStream_t)stream>>>(
      (const double *)src, (float2 *)dst, n);
  return dst;
}
