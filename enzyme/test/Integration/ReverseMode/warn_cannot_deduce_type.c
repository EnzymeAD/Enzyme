// Falling back to defaultTypeTreeForLLVM when TypeAnalysis cannot deduce the
// type of an active load or store is a correctness concern (for instance, an
// untyped i64 swap of doubles under -enzyme-strict-aliasing=0 is assumed to be
// a pointer copy and mirrored forward instead of reversed), so it is reported
// via EmitWarningAlways without requiring -Rpass=enzyme.

// RUN: %clang -std=c11 -g -O2 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -mllvm -enzyme-strict-aliasing=0 2>&1 | FileCheck %s
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o /dev/null %newLoadClangEnzyme -mllvm -enzyme-strict-aliasing=0 2>&1 | FileCheck %s; fi

// CHECK: warning: {{.*}}Enzyme: Cannot deduce type of store
// CHECK: warning: {{.*}}Enzyme: Cannot deduce type of load

extern void __enzyme_autodiff(void *, ...);
extern int enzyme_dup;

void f(void *d) {
  char *c = (char *)d;
  unsigned long x, y;
  __builtin_memcpy(&x, c + 48, sizeof(x));
  __builtin_memcpy(&y, c + 64, sizeof(y));
  __builtin_memcpy(c + 48, &y, sizeof(y));
  __builtin_memcpy(c + 64, &x, sizeof(x));
  ((double *)d)[0] += 1.0;
}

void run(void *d, void *dd) {
  __enzyme_autodiff((void *)f, enzyme_dup, d, dd);
}
