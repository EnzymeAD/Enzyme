#include "common.h"
#include <cstdio>

int main() {
  const int n = 3000, m = 2048;
  double *hx = new double[n], *hy = new double[n];
  for (int i = 0; i < n; ++i) {
    hx[i] = 1.0 + 0.003 * i;
    hy[i] = 0.5 - 0.0007 * i;
  }
  float *hg = new float[9 * m];
  unsigned s = 12345u;
  for (int k = 0; k < 9 * m; ++k) {
    s = s * 1664525u + 1013904223u;
    hg[k] = 0.02f * ((float)(s >> 8) / 16777216.0f - 0.5f);
  }
  double *x, *y, *oa, *ob, *wa, *wb;
  float *g, *va, *vb;
  cudaMalloc(&x, n * sizeof(double));
  cudaMalloc(&y, n * sizeof(double));
  cudaMalloc(&oa, n * sizeof(double));
  cudaMalloc(&ob, n * sizeof(double));
  cudaMalloc(&wa, n * sizeof(double));
  cudaMalloc(&wb, n * sizeof(double));
  cudaMalloc(&g, 9 * m * sizeof(float));
  cudaMalloc(&va, m * sizeof(float));
  cudaMalloc(&vb, m * sizeof(float));
  cudaMemcpy(x, hx, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y, hy, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(g, hg, 9 * m * sizeof(float), cudaMemcpyHostToDevice);
  for (int rep = 0; rep < 2; ++rep) {
    run_a(oa, wa, x, y, g, va, n, m);
    run_b(ob, wb, x, g, vb, n, m);
  }
  double *ha = new double[n], *hb = new double[n];
  double *hwa = new double[n], *hwb = new double[n];
  float *fa = new float[m], *fb = new float[m];
  cudaMemcpy(ha, oa, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb, ob, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(hwa, wa, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(hwb, wb, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fa, va, m * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(fb, vb, m * sizeof(float), cudaMemcpyDeviceToHost);
  double sa = 0, sb = 0, swa = 0, swb = 0, sva = 0, svb = 0;
  for (int i = 0; i < n; ++i) {
    sa += ha[i];
    sb += hb[i];
    swa += hwa[i];
    swb += hwb[i];
  }
  for (int i = 0; i < m; ++i) {
    sva += fa[i];
    svb += fb[i];
  }
  printf("axpb %.17g lift %.17g forall_a %.17g forall_b %.17g vol_a %.9g "
         "vol_b %.9g\n",
         sa, sb, swa, swb, sva, svb);
  cudaError_t e = cudaDeviceSynchronize();
  printf("status %s\n", cudaGetErrorString(e));
  return e != cudaSuccess;
}
