// Mismatched activity is a correctness concern, so it is reported as an
// ordinary warning rather than requiring the user to pass -Rpass=enzyme.

// RUN: %clang -std=c11 -g -O1 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -mllvm -enzyme-postopt=0
// RUN: %clang -std=c11 -g -O2 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -mllvm -enzyme-postopt=0
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O1 %s -S -emit-llvm -o /dev/null %newLoadClangEnzyme -Xclang -verify -mllvm -enzyme-postopt=0; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o /dev/null %newLoadClangEnzyme -Xclang -verify -mllvm -enzyme-postopt=0; fi

// It remains a warning rather than an error, so compilation still succeeds.

// RUN: %clang -std=c11 -g -O2 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -mllvm -enzyme-postopt=0 2>&1 | FileCheck %s
// CHECK: warning: Enzyme: Mismatched activity
// CHECK-SAME: enzyme_runtime_activity

extern void __enzyme_autodiff(void *, ...);
extern int enzyme_const, enzyme_dup;

struct Container {
  double *data;
};

void f(struct Container *c, double *q) {
  c->data = q; // expected-warning-re {{Enzyme: Mismatched activity for:{{.*}}enzyme_runtime_activity{{.*}}}}
}

void run(struct Container *c, struct Container *dc, double *q) {
  __enzyme_autodiff((void *)f, enzyme_dup, c, dc, enzyme_const, q);
}
