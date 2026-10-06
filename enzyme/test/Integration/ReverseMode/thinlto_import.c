// REQUIRES: lld
// Under ThinLTO, lld differentiates each module after the thin link imported
// functions into it. The importer follows calls, so a function that a module
// only passes to __enzyme_autodiff would stay a declaration there. The pre-link
// run asks for it to be imported, together with the functions it calls.

// The pre-link module requests `f` through an anchor function whose profile
// metadata lists its GUID, which the module summary turns into a call edge.
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto=thin %loadClangEnzyme -c %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PRELINK; fi
// PRELINK: @llvm.compiler.used = {{.*}}@enzyme.thinlto.imports
// PRELINK: define internal void @enzyme.thinlto.imports() {{.*}}!prof ![[PROF:[0-9]+]]
// PRELINK: ![[PROF]] = !{!"function_entry_count", i64 1, i64 {{-?[0-9]+}}}

// `f` and `g` each live in their own object file; d/dx (x*x)*x = 3*x^2 = 12.
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto=thin %loadClangEnzyme -c %s -o %t.main.o; fi
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto=thin %loadClangEnzyme -DDEFINE_F -c %s -o %t.f.o; fi
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto=thin %loadClangEnzyme -DDEFINE_G -c %s -o %t.g.o; fi
// RUN: if [ %llvmver -ge 20 ]; then %clang -O2 -flto=thin %linkLLDEnzyme %t.main.o %t.f.o %t.g.o -o %t && %t | FileCheck %s; fi
// CHECK: 12.000000

#if defined(DEFINE_F)
double g(double);
double f(double x) { return g(x) * x; }
#elif defined(DEFINE_G)
double g(double x) { return x * x; }
#else
#include <stdio.h>
double f(double);
double __enzyme_autodiff(void *, double);
int main(void) {
  printf("%f\n", __enzyme_autodiff((void *)f, 2.0));
  return 0;
}
#endif
