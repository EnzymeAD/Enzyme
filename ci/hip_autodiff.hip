// Smoke test of Enzyme on an AMD GPU: differentiates a device function inside HIP kernels, in reverse mode
// (__enzyme_autodiff) and forward mode (__enzyme_fwddiff), and checks both against the analytic derivative.
// Build: clang++ -x hip --offload-arch=<gfx> -O2 -fplugin=ClangEnzyme-<LLVM major>.so hip_autodiff.hip
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                                          \
  do {                                                                                           \
    hipError_t err = (call);                                                                     \
    if (err != hipSuccess) {                                                                     \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(err)); \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

__device__ int enzyme_dup;
__device__ int enzyme_const;

typedef void (*f_ptr)(const double *, double *);
extern __device__ void __enzyme_autodiff(f_ptr, int, const double *, double *, int, double *, double *);
extern __device__ void __enzyme_fwddiff(f_ptr, int, const double *, const double *, int, double *, double *);

// y[i] = sin(x[i]) * x[i]  =>  dy[i]/dx[i] = cos(x[i]) * x[i] + sin(x[i])
__device__ void f(const double *x, double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  y[i] = sin(x[i]) * x[i];
}

// Reverse mode: dx += (dy/dx)^T dy, with dy = 1 (dy is zeroed by the reverse pass).
__global__ void grad_f(const double *x, double *dx, double *y, double *dy) {
  __enzyme_autodiff(f, enzyme_dup, x, dx, enzyme_dup, y, dy);
}

// Forward mode: dy = (dy/dx) dx, with dx = 1.
__global__ void tangent_f(const double *x, const double *dx, double *y, double *dy) {
  __enzyme_fwddiff(f, enzyme_dup, x, dx, enzyme_dup, y, dy);
}

int main() {
  const int n = 1024, threads = 256, blocks = n / threads;
  const size_t bytes = n * sizeof(double);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);

  double *x = new double[n], *dx = new double[n], *y = new double[n], *dy = new double[n];
  for (int i = 0; i < n; ++i) x[i] = -3.0 + 6.0 * i / (n - 1);

  double *d_x, *d_dx, *d_y, *d_dy;
  HIP_CHECK(hipMalloc(&d_x, bytes));
  HIP_CHECK(hipMalloc(&d_dx, bytes));
  HIP_CHECK(hipMalloc(&d_y, bytes));
  HIP_CHECK(hipMalloc(&d_dy, bytes));
  HIP_CHECK(hipMemcpy(d_x, x, bytes, hipMemcpyHostToDevice));

  auto check = [&](const char *mode, const double *d) {
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i)
      maxerr = std::fmax(maxerr, std::fabs(d[i] - (std::cos(x[i]) * x[i] + std::sin(x[i]))));
    bool ok = maxerr < 1e-12;
    std::printf("%-8s mode: max |error| = %.3e  %s\n", mode, maxerr, ok ? "PASSED" : "FAILED");
    return ok;
  };

  // Reverse mode
  for (int i = 0; i < n; ++i) { dx[i] = 0.0; dy[i] = 1.0; }
  HIP_CHECK(hipMemcpy(d_dx, dx, bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_dy, dy, bytes, hipMemcpyHostToDevice));
  grad_f<<<blocks, threads>>>(d_x, d_dx, d_y, d_dy);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(dx, d_dx, bytes, hipMemcpyDeviceToHost));
  bool ok = check("reverse", dx);

  // Forward mode
  for (int i = 0; i < n; ++i) dx[i] = 1.0;
  HIP_CHECK(hipMemcpy(d_dx, dx, bytes, hipMemcpyHostToDevice));
  tangent_f<<<blocks, threads>>>(d_x, d_dx, d_y, d_dy);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(dy, d_dy, bytes, hipMemcpyDeviceToHost));
  ok = check("forward", dy) && ok;

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_dx));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_dy));
  delete[] x; delete[] dx; delete[] y; delete[] dy;
  return ok ? 0 : 1;
}
