// clang-format off
// RUN: %clang -std=c11 -O2 -DSTATIC_MARKERS %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O3 -DSTATIC_MARKERS %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: if [ %llvmver -ge 16 ]; then %clang -std=c11 %O0TBAA -flto -c %s -emit-llvm -o %t.bc && %opt %loadLLDEnzymeOpt -passes="internalize,lto<O2>" -internalize-public-api-list=main %t.bc -S -o - | FileCheck %s; fi
// RUN: if [ %llvmver -ge 16 ]; then %clang -std=c11 %O0TBAA -flto -c %s -emit-llvm -o %t.bc && %opt %loadLLDEnzymeOpt -passes="internalize,lto<O2>" -internalize-public-api-list=main %t.bc -S -o - | %lli -; fi
// RUN: if [ %llvmver -ge 16 ]; then %clang -std=c11 %O0TBAA -flto=thin -c %s -emit-llvm -o %t.thin.bc && %opt %loadLLDEnzymeOpt -passes="internalize,thinlto<O2>" -internalize-public-api-list=main %t.thin.bc -S -o - | FileCheck %s; fi
// RUN: if [ %llvmver -ge 16 ]; then %clang -std=c11 %O0TBAA -flto=thin -c %s -emit-llvm -o %t.thin.bc && %opt %loadLLDEnzymeOpt -passes="internalize,thinlto<O2>" -internalize-public-api-list=main %t.thin.bc -S -o - | %lli -; fi
// clang-format on

// CHECK-NOT: @enzyme_
// CHECK-NOT: __enzyme_autodiff

#ifdef STATIC_MARKERS
#define MARKER_STORAGE static
#else
#define MARKER_STORAGE
#endif

MARKER_STORAGE int enzyme_dup;
MARKER_STORAGE int enzyme_dupnoneed;
MARKER_STORAGE int enzyme_out;
MARKER_STORAGE int enzyme_const;

extern void abort(void);
void __enzyme_autodiff(void *, ...);

static double logpdf(const double *x, const double *mu, const double *std,
                     double scale) {
  double z = (*x - *mu) / *std;
  return (-0.5 * z * z - __builtin_log(*std)) * scale;
}

static void grad_logpdf(double x, double mu, double std, double scale,
                        double *diffs) {
  __enzyme_autodiff((void *)logpdf, enzyme_dup, &x, &diffs[0], enzyme_dup, &mu,
                    &diffs[1], enzyme_dup, &std, &diffs[2], enzyme_const,
                    scale);
}

int main(void) {
  double diffs[3] = {0.0, 0.0, 0.0};
  grad_logpdf(0.5, 0.0, 1.0, 1.0, diffs);
  if (__builtin_fabs(diffs[0] - (-0.5)) > 1e-12)
    abort();
  if (__builtin_fabs(diffs[1] - 0.5) > 1e-12)
    abort();
  if (__builtin_fabs(diffs[2] - (-0.75)) > 1e-12)
    abort();
  return 0;
}
