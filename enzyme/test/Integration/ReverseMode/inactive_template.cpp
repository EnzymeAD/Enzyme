// Enzyme's registration attributes are lowered to a global holding the address
// of the annotated declaration, which does not exist until a template is
// instantiated. Check that they nonetheless apply to declarations written
// inside templates.

// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++14 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++14 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++14 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++14 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++14 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include <stdio.h>

#include "../test_utils.h"

double __enzyme_autodiff(void *, ...);

// A function template.
template <typename T>
__attribute__((noinline)) __attribute__((enzyme_inactive)) T inactive_fn(T a) {
  return 2 * a;
}

// A member function of a class template, and a static data member of one.
template <typename T> struct Holder {
  __attribute__((noinline)) __attribute__((enzyme_inactive)) static T
  member_fn(T a) {
    return 3 * a;
  }

  __attribute__((enzyme_inactive)) static T tally;

  __attribute__((noinline)) static void accumulate(T a, T *c) {
    tally += a;
    *c *= a;
  }
};
template <typename T> T Holder<T>::tally = 0;

// A variable template.
template <typename T> __attribute__((enzyme_inactive)) T inactive_var = 0;

template <typename T>
__attribute__((noinline)) void accumulate_var(T a, T *c) {
  inactive_var<T> += a;
  *c *= a;
}

// The two inactive calls contribute nothing to the derivative, and the stores
// into the two inactive globals carry none either, leaving d(a*a)/da = 2a.
double test(double a) {
  double dat = 1.0;
  Holder<double>::accumulate(a, &dat);
  accumulate_var<double>(a, &dat);
  return dat + inactive_fn<double>(a) + Holder<double>::member_fn(a);
}

int main(int argc, char **argv) {
  double out = __enzyme_autodiff((void *)test, 3.0);
  printf("out=%f\n", out);
  APPROX_EQ(out, 6.0, 1e-10);

  // The inactive globals are written by the primal only, never by the reverse
  // pass.
  APPROX_EQ(Holder<double>::tally, 3.0, 1e-10);
  APPROX_EQ(inactive_var<double>, 3.0, 1e-10);
  return 0;
}
