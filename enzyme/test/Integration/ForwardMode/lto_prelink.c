// An LTO pre-link pipeline only ever sees one translation unit, so the function
// named by __enzyme_fwddiff may still be a bare declaration. Enzyme defers to
// the post-link run instead of failing here; `square` below stands in for a
// callee whose body lives in another object file.

// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto %loadClangEnzyme -c %s -o /dev/null; fi
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto=thin %loadClangEnzyme -c %s -o /dev/null; fi
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O0 -flto %loadClangEnzyme -c %s -o /dev/null; fi

// Once the definition is linked in, the post-link pipeline differentiates the
// call that the pre-link run left alone: d/dx x*x == 2*x.
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto %loadClangEnzyme -c %s -emit-llvm -o %t.caller.bc && %clang -std=c11 -O2 -flto -DDEFINITION -c %s -emit-llvm -o %t.callee.bc && llvm-link %t.caller.bc %t.callee.bc -o %t.bc && %opt %optLoadClangEnzyme -passes="lto<O2>" %t.bc -S -o - | FileCheck %s; fi

// CHECK-LABEL: @dsquare(
// CHECK-NOT: __enzyme_fwddiff
// CHECK: fmul {{.*}}double %{{.*}}, 2.000000e+00

// -enzyme-lto-prelink=1 restores the old behaviour, which cannot resolve the
// callee from this translation unit alone.
// RUN: if [ %llvmver -ge 20 ]; then not %clang -std=c11 -O2 -flto %loadClangEnzyme -mllvm -enzyme-lto-prelink=1 -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNRESOLVED; fi

// Without LTO nothing changes: Enzyme still runs per translation unit, and the
// declaration is still an error there.
// RUN: not %clang -std=c11 -O2 %loadClangEnzyme -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNRESOLVED

// UNRESOLVED: Enzyme: failed to find fn to differentiate

#ifdef DEFINITION
double square(double x) { return x * x; }
#else
extern double square(double x);

double __enzyme_fwddiff(void *, double, double);

double dsquare(double x) { return __enzyme_fwddiff((void *)square, x, 1.0); }
#endif
