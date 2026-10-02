// ozaki_calibrate: measure the Ozaki-II host-dispatch cost relative to the
// naive scalar-FP64 GEMM, for the cost model, which prices a host-dispatch
// candidate as
//     compCost = FP64_baseline_per_MAC x rel(nm) x padWaste(S^3/MNK),
// rel(nm) = (dispatch time) / (scalar-FP64 GEMM time) at a square reference
// shape per modulus count. Emits one row per modulus count the candidate
// generator can propose (nm 8..14) plus the native cuBLAS DGEMM row
// `ozaki_dispatch_rel,dgemm,<rel>` the native DGEMM candidate is priced from.
// Build: nvcc -O3 -arch=sm_120 ozaki_calibrate.cu <path>/ozaki_rt.cu \
//        -lcublas -o ozaki_calibrate
// Run:   ./ozaki_calibrate [N=2048] [reps=30]  >>  cm_sm_120_RTX5090.csv
//        (strip any prior `ozaki_dispatch_rel,` rows from the CSV before appending)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Square host dispatch (the entry ozp/dm_purify use for square GEMMs).
extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int N, int lda, int ldb,
                                       int ldc, int transA, int transB,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli);

#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// Naive scalar-FP64 GEMM = the cost model's FP64 baseline (per-MAC throughput
// of a triple-loop reduction); this is the denominator rel is measured against.
__global__ void ref_gemm(double *C, const double *A, const double *B, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= N) return;
  double s = 0.0;
  for (int l = 0; l < N; ++l) s += A[(size_t)row * N + l] * B[(size_t)col * N + l];
  C[(size_t)col * N + row] = s;
}

static int N = 2048, REPS = 30, NM = 14;
static double *dA, *dB, *dC;

static void run_ref() { dim3 b(16,16), g((N+15)/16,(N+15)/16); ref_gemm<<<g,b>>>(dC,dA,dB,N); }
static void run_oz()  { __poseidon_ozaki_dgemm(dC,dA,dB,N,N,N,N,0,0,1.0,0.0,0,NM); }

static float timeit(void (*fn)(), int reps) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  fn(); CK(cudaDeviceSynchronize());                       // warmup
  cudaEventRecord(a); for (int i=0;i<reps;i++) fn(); cudaEventRecord(b);
  cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b); return ms/reps;
}

int main(int argc, char **argv) {
  if (argc > 1) N = atoi(argv[1]);
  if (argc > 2) REPS = atoi(argv[2]);
  size_t sz = (size_t)N * N;
  double *hA = (double*)malloc(sz*8), *hB = (double*)malloc(sz*8);
  srand(1);
  for (size_t i=0;i<sz;i++) { hA[i]=((double)rand()/RAND_MAX)*2-1; hB[i]=((double)rand()/RAND_MAX)*2-1; }
  CK(cudaMalloc(&dA,sz*8)); CK(cudaMalloc(&dB,sz*8)); CK(cudaMalloc(&dC,sz*8));
  CK(cudaMemcpy(dA,hA,sz*8,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB,sz*8,cudaMemcpyHostToDevice));

  float t_ref = timeit(run_ref, REPS);
  fprintf(stderr, "[ozaki_calibrate] N=%d reps=%d  scalar-FP64 ref = %.4f ms\n",
          N, REPS, t_ref);

  // Every modulus count the dispatch candidate generator can propose: a
  // candidate whose `ozaki_dispatch_rel` row is missing would be priced from
  // the analytic estimate. Re-measured in one session so the rungs are
  // mutually consistent.
  const int moduli[] = {8, 9, 10, 11, 12, 13, 14};
  const int nModuli = (int)(sizeof(moduli) / sizeof(moduli[0]));
  for (int mi = 0; mi < nModuli; ++mi) {
    NM = moduli[mi];
    float t_oz = timeit(run_oz, REPS);
    double rel = (double)t_oz / (double)t_ref;
    fprintf(stderr, "[ozaki_calibrate] nm=%2d  dispatch = %.4f ms  rel = %.4f "
                    "(%.2fx vs FP64)\n", NM, t_oz, rel, t_ref/t_oz);
    // Clean CSV row on stdout for appending to the cost model.
    printf("ozaki_dispatch_rel,nm%d,%.6f\n", NM, rel);
  }

  // Native cuBLAS DGEMM through the same dispatch entry (nm=0), exactly what
  // the native DGEMM candidate runs; handle creation amortized by the warmup
  // call.
  NM = 0;
  float t_dg = timeit(run_oz, REPS);
  double rel_dg = (double)t_dg / (double)t_ref;
  fprintf(stderr, "[ozaki_calibrate] dgemm  dispatch = %.4f ms  rel = %.4f "
                  "(%.2fx vs FP64)\n", t_dg, rel_dg, t_ref/t_dg);
  printf("ozaki_dispatch_rel,dgemm,%.6f\n", rel_dg);
  return 0;
}
