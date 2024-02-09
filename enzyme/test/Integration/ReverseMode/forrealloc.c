// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %loadClangEnzyme %s -S -emit-llvm -o - | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %loadClangEnzyme %s -S -emit-llvm -o - | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %loadClangEnzyme %s -S -emit-llvm -o - -mllvm -enzyme-loose-types | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %loadClangEnzyme %s -S -emit-llvm -o - -mllvm -enzyme-loose-types | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %loadClangEnzyme %s -S -emit-llvm -o - -mllvm -enzyme-inline=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %loadClangEnzyme %s -S -emit-llvm -o - -mllvm -enzyme-inline=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %loadClangEnzyme %s -S -emit-llvm -o - -mllvm -enzyme-inline=1 -mllvm -enzyme-loose-types | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %loadClangEnzyme %s -S -emit-llvm -o - -mllvm -enzyme-inline=1 -mllvm -enzyme-loose-types | %lli - ; fi

#include "../test_utils.h"

float __enzyme_autodiff(void*, float, int);

float foo(float inp, int n) {
  float* x = 0;
  for(int i=0; i<n; i++) {
    x = (float*)realloc(x, (i+1)*sizeof(float));
    if (i == 0 ){
        *x = inp;
    } else {
        x[i] = x[i-1] + inp;
    }
  }
  float res = x[n-1];
  free(x);
  return res;
}


int main(int argc, char** argv) {
  float inp = 3.0f;
  float res = __enzyme_autodiff(foo, inp, 32);

  printf("hello! inp=%f, res=%f\n", inp, res);
  APPROX_EQ(res, 32.0f, 1e-10);

  return 0;
}
