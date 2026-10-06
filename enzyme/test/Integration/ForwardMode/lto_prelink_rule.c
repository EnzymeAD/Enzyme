// A pre-link pipeline that defers differentiation still must not export the
// functions PreserveNVVM made external: every object file including this rule
// would define its own strong `square_`, and linking two of them fails. Their
// original linkage comes back, and llvm.compiler.used keeps them alive instead.

// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto %loadClangEnzyme -c %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PRELINK; fi
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto=thin %loadClangEnzyme -c %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PRELINK; fi

// PRELINK: @llvm.compiler.used = appending global [2 x ptr] [ptr @square_, ptr @dsquare_]
// PRELINK-DAG: define internal void @square_(
// PRELINK-DAG: call double @__enzyme_fwddiff(
// PRELINK-DAG: define internal void @dsquare_(

// Two translation units with their own copy of the rule link, and post-link
// each call uses its rule. The llvm.compiler.used entries go away again.
// RUN: if [ %llvmver -ge 20 ]; then %clang -std=c11 -O2 -flto %loadClangEnzyme -c %s -emit-llvm -o %t.a.bc && %clang -std=c11 -O2 -flto %loadClangEnzyme -DSECOND -c %s -emit-llvm -o %t.b.bc && llvm-link %t.a.bc %t.b.bc -o %t.bc && %opt %loadLLDEnzymeOpt -passes="lto<O2>" %t.bc -S -o - | FileCheck %s; fi

// CHECK-NOT: llvm.compiler.used
// CHECK-LABEL: define {{.*}} double @dsquare_a(
// CHECK-NEXT: ret double 1.000000e+02
// CHECK-LABEL: define {{.*}} double @dsquare_b(
// CHECK-NEXT: ret double 1.000000e+02
// CHECK-NOT: __enzyme_fwddiff

static void square_(const double *src, double *dest) { *dest = *src * *src; }

static void dsquare_(const double *src, const double *d_src,
                     const double *dest, double *d_dest) {
  // intentionally incorrect, to tell the rule apart from Enzyme's derivative
  *d_dest = 100;
}

void *__enzyme_register_derivative_square[] = {
    (void *)square_,
    (void *)dsquare_,
};

static double square(double x) {
  double y;
  square_(&x, &y);
  return y;
}

double __enzyme_fwddiff(void *, double, double);

#ifdef SECOND
double dsquare_b(double x) { return __enzyme_fwddiff((void *)square, x, 1.0); }
#else
double dsquare_a(double x) { return __enzyme_fwddiff((void *)square, x, 1.0); }
#endif
