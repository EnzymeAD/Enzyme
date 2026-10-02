// direct_calibrate: measure the direct reduced-precision GEMM host-dispatch
// cost relative to the naive scalar-FP64 GEMM, for the cost model:
//     rel = (dispatch time) / (scalar-FP64 GEMM time)
// at a square reference shape, the same protocol as ozaki_calibrate.cu and
// tcec_calibrate.cu so the three families are priced in one unit. The narrow
// and widen passes are inside the timed region. Emits one row per operand
// format (matmul/MatmulCandidates.cpp kDirectDispatchVariants):
//     direct_dispatch_rel,f16_f32,<rel>
//     direct_dispatch_rel,bf16_f32,<rel>
//     direct_dispatch_rel,tf32_f32,<rel>
//     direct_dispatch_rel,f32_f32,<rel>   (cuBLAS SGEMM, no tensor cores)
// Build: nvcc -O3 -arch=sm_120 direct_calibrate.cu <path>/direct_rt.cu \
//        -lcublas -o direct_calibrate
// Run:   ./direct_calibrate [N=2048] [reps=30]   (CSV rows on stdout)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" void __poseidon_direct_dgemm(double *C, const double *A,
                                        const double *B, int N, int lda,
                                        int ldb, int ldc, int transA,
                                        int transB, double alpha, double beta,
                                        cudaStream_t stream, int mode);

#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// Byte-identical baseline kernel to ozaki_calibrate.cu's and
// tcec_calibrate.cu's ref_gemm: the three rels must share a denominator or the
// families are not comparable.
__global__ void ref_gemm(double *C, const double *A, const double *B, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= N) return;
  double s = 0.0;
  for (int l = 0; l < N; ++l) s += A[(size_t)row * N + l] * B[(size_t)col * N + l];
  C[(size_t)col * N + row] = s;
}

static int N = 2048, REPS = 30, MODE = 1;
static double *dA, *dB, *dC;

static void run_ref() { dim3 b(16,16), g((N+15)/16,(N+15)/16); ref_gemm<<<g,b>>>(dC,dA,dB,N); }
static void run_dr()  { __poseidon_direct_dgemm(dC,dA,dB,N,N,N,N,0,0,1.0,0.0,0,MODE); }

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
  fprintf(stderr, "[direct_calibrate] N=%d reps=%d  scalar-FP64 ref = %.4f ms\n",
          N, REPS, t_ref);

  // Same order and the same class spelling the cost-model rows use.
  const int modes[] = {1, 2, 3, 4};
  const char *classes[] = {"f16_f32", "bf16_f32", "tf32_f32", "f32_f32"};
  for (int i = 0; i < 4; ++i) {
    MODE = modes[i];
    float t = timeit(run_dr, REPS);
    double rel = (double)t / (double)t_ref;
    fprintf(stderr, "[direct_calibrate] %-9s dispatch = %.4f ms  rel = %.6f "
                    "(%.2fx vs FP64)\n", classes[i], t, rel, t_ref/t);
    printf("direct_dispatch_rel,%s,%.6f\n", classes[i], rel);
  }
  return 0;
}
