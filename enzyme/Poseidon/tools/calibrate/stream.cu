//===- stream.cu - price streamed memory traffic --------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The stream rows of a GPU cost model, measured on the GPU it runs on: built
// as CUDA (poseidon-calibrate-stream-cuda) and as HIP
// (poseidon-calibrate-stream-hip). The price of an element a kernel reads or
// writes from an array far larger than the last-level cache, in the unit of the
// per-operation rows (machine-wide nanoseconds per element x clock in GHz x 2,
// as ops_gpu.cu), a store priced as what it adds to a kernel that reads as
// much as it writes, and how the arithmetic and the memory time of one kernel
// combine:
//     stream_load,<type>,<cost>    stream_store,<type>,<cost>
//     stream_combine,pnorm,<p>     stream_combine,band,<b>
// The combination is fitted to kernels that read an element, apply k dependent
// fused multiply-adds and write it, for k from 0 until the arithmetic is many
// times the memory time: T = (A^p + M^p)^(1/p) with A = k x fmuladd row and
// M = load + store row. band is the largest relative deviation of a measured
// time from the fitted model where A <= M. The fit table goes to stderr.
// Usage: poseidon-calibrate-stream-cuda <fmuladd,double> <fmuladd,float>; rows
// on stdout.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/DeviceApi.h"
#include "../../runtime/profile/ProfileFormat.h"
#include "gpu_timing.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Independent elements in flight per thread: 32 bytes of each type.
template <typename T> constexpr int kIlp = 32 / sizeof(T);
constexpr int kThreads = 256;
constexpr int kReps = 10;

__device__ inline double fmaOp(double v, double a, double b) {
  return fma(v, a, b);
}
__device__ inline float fmaOp(float v, float a, float b) {
  return fmaf(v, a, b);
}
__device__ inline unsigned long long bitsOf(double v) {
  return (unsigned long long)__double_as_longlong(v);
}
__device__ inline unsigned bitsOf(float v) { return __float_as_uint(v); }

// n is a multiple of kIlp<T> x the grid's thread count.
template <typename T> __global__ void readOnly(const T *x, T *sink, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  decltype(bitsOf(T())) acc = 0;
  for (size_t i = tid; i < n; i += kIlp<T> * stride) {
#pragma unroll
    for (int u = 0; u < kIlp<T>; ++u)
      acc ^= bitsOf(x[i + u * stride]);
  }
  if (acc == 1)
    sink[tid] = T(1);
}

template <typename T> __global__ void writeOnly(T *y, size_t n, T v0) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const T v = v0 + (T)threadIdx.x;
  for (size_t i = tid; i < n; i += kIlp<T> * stride) {
#pragma unroll
    for (int u = 0; u < kIlp<T>; ++u)
      y[i + u * stride] = v;
  }
}

template <typename T, int K>
__global__ void transform(const T *x, T *y, size_t n, T a, T b) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < n; i += kIlp<T> * stride) {
    T v[kIlp<T>];
#pragma unroll
    for (int u = 0; u < kIlp<T>; ++u)
      v[u] = x[i + u * stride];
#pragma unroll 16
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int u = 0; u < kIlp<T>; ++u)
        v[u] = fmaOp(v[u], a, b);
#pragma unroll
    for (int u = 0; u < kIlp<T>; ++u)
      y[i + u * stride] = v[u];
  }
}

static int gBlocks = 0;

// The price of one element of a kernel launched over n and 2n elements.
template <typename F> static double perElement(size_t n, F launchOver) {
  double t1 = gpu_timing::timeNs([&] { launchOver(n); }, kReps);
  double t2 = gpu_timing::timeNs([&] { launchOver(2 * n); }, kReps);
  return gpu_timing::cyclesPerUnit(t1, t2, (double)n);
}

// Prints the row `op`,`type`. A price that is not positive measured nothing,
// and no row is written in its place.
static void printRow(const char *op, const char *type, double price) {
  if (!(price > 0))
    poseidon::fatal("%s,%s: the measured price %.9g is not positive, so the "
                    "row has no price",
                    op, type, price);
  printf("%s,%s,%.9g\n", op, type, price);
}

struct Point {
  const char *type;
  int k;
  double a, m, t;
};

template <typename T, int K>
static void sweepOne(std::vector<Point> &pts, const char *name, const T *x,
                     T *y, size_t n, double fma, double m) {
  double t = perElement(n, [&](size_t len) {
    transform<T, K><<<gBlocks, kThreads>>>(x, y, len, T(0.999), T(0.001));
  });
  pts.push_back({name, K, K * fma, m, t});
}

// k dependent operations per element for k = 0, 1, 2, 4, ... until the
// arithmetic is sixteen times the memory time.
template <typename T>
static void sweep(std::vector<Point> &pts, const char *name, const T *x, T *y,
                  size_t n, double fma, double m) {
  sweepOne<T, 0>(pts, name, x, y, n, fma, m);
#define POSEIDON_SWEEP(K)                                                      \
  if ((K) / 2 * fma < 16 * m)                                                  \
    sweepOne<T, K>(pts, name, x, y, n, fma, m);
  POSEIDON_SWEEP(1)
  POSEIDON_SWEEP(2)
  POSEIDON_SWEEP(4)
  POSEIDON_SWEEP(8)
  POSEIDON_SWEEP(16)
  POSEIDON_SWEEP(32)
  POSEIDON_SWEEP(64)
  POSEIDON_SWEEP(128)
  POSEIDON_SWEEP(256)
  POSEIDON_SWEEP(512)
  POSEIDON_SWEEP(1024)
  POSEIDON_SWEEP(2048)
  POSEIDON_SWEEP(4096)
#undef POSEIDON_SWEEP
}

static double misfit(const std::vector<Point> &pts, double p) {
  double s = 0.0;
  for (const Point &q : pts) {
    double r = std::log(q.t / poseidon::combineTimes(q.a, q.m, p));
    s += r * r;
  }
  return s;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <fmuladd,double> <fmuladd,float>\n", argv[0]);
    return 2;
  }
  const double fmaD = atof(argv[1]), fmaF = atof(argv[2]);
  if (!(fmaD > 0.0) || !(fmaF > 0.0)) {
    fprintf(stderr, "%s: the fmuladd rows must be positive\n", argv[0]);
    return 2;
  }
  gpuDeviceProp_t prop;
  POSEIDON_GPU_CHECK(gpuGetDeviceProperties(&prop, 0));
  // A clock rate that is not positive stops the run before any row is timed.
  gpu_timing::clockGhz();
  gBlocks =
      prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / kThreads);
  const size_t grid = (size_t)gBlocks * kThreads * kIlp<float>;
  // 2^27 elements and twice that, rounded to whole grid sweeps: 1 GiB of
  // doubles, far beyond any last-level cache.
  const size_t n = ((size_t)1 << 27) / grid * grid;

  double *xd, *yd;
  POSEIDON_GPU_CHECK(gpuMalloc(&xd, 2 * n * sizeof(double)));
  POSEIDON_GPU_CHECK(gpuMalloc(&yd, 2 * n * sizeof(double)));
  POSEIDON_GPU_CHECK(gpuMemset(xd, 0x3f, 2 * n * sizeof(double)));
  POSEIDON_GPU_CHECK(gpuMemset(yd, 0, 2 * n * sizeof(double)));
  float *xf = (float *)xd, *yf = (float *)yd;

  // Bring the GPU to its sustained clock before the first row.
  for (int i = 0; i < 20; ++i)
    transform<double, 8><<<gBlocks, kThreads>>>(xd, yd, 2 * n, 0.999, 0.001);
  POSEIDON_GPU_CHECK(gpuDeviceSynchronize());

  double loadD = perElement(n, [&](size_t len) {
    readOnly<double><<<gBlocks, kThreads>>>(xd, yd, len);
  });
  double loadF = perElement(n, [&](size_t len) {
    readOnly<float><<<gBlocks, kThreads>>>(xf, yf, len);
  });
  double writeD = perElement(n, [&](size_t len) {
    writeOnly<double><<<gBlocks, kThreads>>>(yd, len, 1.5);
  });
  double writeF = perElement(n, [&](size_t len) {
    writeOnly<float><<<gBlocks, kThreads>>>(yf, len, 1.5f);
  });
  // A kernel that writes also reads, and reads and writes to the same memory
  // share it less well than either alone: a store is priced as what it adds
  // to a copy.
  double copyD = perElement(n, [&](size_t len) {
    transform<double, 0><<<gBlocks, kThreads>>>(xd, yd, len, 0.999, 0.001);
  });
  double copyF = perElement(n, [&](size_t len) {
    transform<float, 0><<<gBlocks, kThreads>>>(xf, yf, len, 0.999f, 0.001f);
  });
  const double storeD = copyD - loadD, storeF = copyF - loadF;

  std::vector<Point> pts;
  sweep<double>(pts, "double", xd, yd, n, fmaD, loadD + storeD);
  sweep<float>(pts, "float", xf, yf, n, fmaF, loadF + storeF);
  POSEIDON_GPU_CHECK(gpuFree(xd));
  POSEIDON_GPU_CHECK(gpuFree(yd));

  // Golden-section search for the exponent on log p in [0, log 64].
  const double g = (std::sqrt(5.0) - 1.0) / 2.0;
  double lo = 0.0, hi = std::log(64.0);
  double c = hi - g * (hi - lo), d = lo + g * (hi - lo);
  for (int it = 0; it < 100; ++it) {
    if (misfit(pts, std::exp(c)) < misfit(pts, std::exp(d)))
      hi = d;
    else
      lo = c;
    c = hi - g * (hi - lo);
    d = lo + g * (hi - lo);
  }
  const double p = std::exp((lo + hi) / 2.0);

  // The band is the largest deviation where the memory time is at least the
  // arithmetic time: those points test the memory prices and the
  // combination. Beyond, the sweep tests the per-operation rows, whose
  // deviation is reported but is not the combination's.
  double band = 0.0, beyond = 0.0, rms = 0.0;
  fprintf(
      stderr,
      "poseidon-calibrate-stream: %s, %d blocks x %d threads, %zu elements, "
      "%.3f GHz\n",
      prop.name, gBlocks, kThreads, n, gpu_timing::clockGhz());
  fprintf(stderr,
          "poseidon-calibrate-stream: double: read %.6g, write alone %.6g, "
          "copy %.6g; float: read %.6g, write alone %.6g, copy %.6g\n",
          loadD, writeD, copyD, loadF, writeF, copyF);
  fprintf(stderr, "poseidon-calibrate-stream: type k A M T_measured T_model "
                  "relative_residual\n");
  for (const Point &q : pts) {
    double model = poseidon::combineTimes(q.a, q.m, p);
    double r = q.t / model - 1.0;
    if (q.a <= q.m)
      band = std::fmax(band, std::fabs(r));
    else
      beyond = std::fmax(beyond, std::fabs(r));
    rms += r * r;
    fprintf(stderr,
            "poseidon-calibrate-stream: %s %d %.6g %.6g %.6g %.6g %+.4f\n",
            q.type, q.k, q.a, q.m, q.t, model, r);
  }
  rms = std::sqrt(rms / pts.size());
  fprintf(
      stderr,
      "poseidon-calibrate-stream: fit p = %.4f over %zu points, rms relative "
      "residual %.4f; largest where A <= M %.4f (the band), where A > M "
      "%.4f\n",
      p, pts.size(), rms, band, beyond);

  printRow("stream_load", "double", loadD);
  printRow("stream_load", "float", loadF);
  printRow("stream_store", "double", storeD);
  printRow("stream_store", "float", storeF);
  printf("stream_combine,pnorm,%.9g\n", p);
  printf("stream_combine,band,%.9g\n", band);
  return 0;
}
