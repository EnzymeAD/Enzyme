//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 -disable-llvm-optzns %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
//   note not doing O0 below as to ensure we get tbaa
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 -Xclang -disable-llvm-optzns %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi
// RUN: if [ %llvmver -ge 9 ]; then %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out ; fi

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "../test_utils.h"

float __enzyme_fwddiff(void*, ...);

float omp(float* a, int N) {
  float res = 0.0;

  #pragma omp parallel
  {
    double thread_res = 0.0;
    
    #pragma omp for
    for (int i=0; i<N; i++) {
      thread_res += a[i] * a[i];
    }

    #pragma omp critical
    {
      res += thread_res;
    }

  }

  return res;
}

int main(int argc, char** argv) {

  int N = 20;
  float a[N];
  for(int i=0; i<N; i++) {
    a[i] = i+1;
  }

  float d_a[N];
  for(int i=0; i<N; i++)
    d_a[i] = 0.0f;

  d_a[N-1] = 1.0;
  
  //omp(*a, N);
  printf("ran omp\n");
  float dret = __enzyme_fwddiff((void*)omp, a, d_a, N);
  printf("dret=%f\n", dret);

    APPROX_EQ(dret, 40.0 , 1e-10);

  return 0;
}
