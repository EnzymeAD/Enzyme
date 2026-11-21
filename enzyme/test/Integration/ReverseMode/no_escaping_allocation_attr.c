// The enzyme_no_escaping_allocation attribute marks a function, whether it is
// defined in this translation unit or only declared, with the
// enzyme_no_escaping_allocation function attribute.

// RUN: if [ %llvmver -ge 11 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | FileCheck %s; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | FileCheck %s; fi

__attribute__((enzyme_no_escaping_allocation)) void external_check(double *);

__attribute__((enzyme_no_escaping_allocation, noinline)) void
local_check(double *x) {
  external_check(x);
}

void unmarked(double *);

void use(double *x) {
  local_check(x);
  unmarked(x);
}

// CHECK-NOT: __enzyme_no_escaping_allocation
// CHECK-DAG: declare void @external_check({{.*}}) {{.*}}[[EXT:#[0-9]+]]
// CHECK-DAG: define {{.*}}void @local_check({{.*}}) {{.*}}[[LOCAL:#[0-9]+]] {
// CHECK-DAG: declare void @unmarked({{.*}}) {{.*}}[[UNMARKED:#[0-9]+]]
// CHECK-DAG: attributes [[EXT]] = { {{.*}}"enzyme_no_escaping_allocation"
// CHECK-DAG: attributes [[LOCAL]] = { {{.*}}"enzyme_no_escaping_allocation"
// CHECK-DAG: attributes [[UNMARKED]] = { {{[^"]*}}"no-trapping-math"
