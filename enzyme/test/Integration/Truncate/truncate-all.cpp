// clang-format off
// Baseline

// RUN: if [ %llvmver -ge 12 ]; then %clang -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -S -mllvm --enzyme-truncate-all="" | %lli - | FileCheck --check-prefix BASELINE %s; fi
// BASELINE: 900000000.560000


// Truncated

// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -include enzyme/fprt/mpfr.h -O3 %s -o %s.a.out %newLoadClangEnzyme -mllvm --enzyme-truncate-all="64to32" -lmpfr -lm &&  %s.a.out | FileCheck --check-prefix TO_32 %s; fi
// TO_32: 900000000.000000

// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -include enzyme/fprt/mpfr.h -O3 %s -o %s.a.out %newLoadClangEnzyme -mllvm --enzyme-truncate-all="11-52to8-23" -lmpfr -lm &&  %s.a.out | FileCheck --check-prefix TO_28_23 %s; fi
// TO_28_23: 900000000.000000

// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -include enzyme/fprt/mpfr.h -O3 %s -o %s.a.out %newLoadClangEnzyme -mllvm --enzyme-truncate-all="11-52to3-7" -lmpfr -lm &&  %s.a.out | FileCheck --check-prefix TO_3_7 %s; fi
// TO_3_7: 897581056.000000

// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -g -include enzyme/fprt/mpfr-test.h -O3 %s -o %s.a.out %newLoadClangEnzyme -mllvm --enzyme-truncate-all="11-52to3-7" -lmpfr -lm &&  %s.a.out | FileCheck --check-prefix CHECK-LOCS %s; fi
// CHECK-LOCS:      0x[[op1:[0-9a-f]*]], {{.*}}truncate-all.cpp:[[op1loc:.*]]
// CHECK-LOCS-NEXT: 0x[[op2:[0-9a-f]*]], {{.*}}truncate-all.cpp:[[op2loc:.*]]
// CHECK-LOCS-NEXT: 0x[[op3:[0-9a-f]*]], {{.*}}truncate-all.cpp:[[op3loc:.*]]
// CHECK-LOCS-NEXT: 0x[[op4:[0-9a-f]*]], {{.*}}truncate-all-header.h:[[op4loc:.*]]
// CHECK-LOCS-NEXT: 0x[[op5:[0-9a-f]*]], {{.*}}truncate-all-header.h:[[op5loc:.*]]
// CHECK-LOCS-NEXT: 0x[[op6:[0-9a-f]*]], {{.*}}truncate-all-header.h:[[op6loc:.*]]
// CHECK-LOCS-NEXT: 0x[[op7:[0-9a-f]*]], {{.*}}truncate-all.cpp:[[op7loc:.*]]
// CHECK-LOCS-NEXT: 0x[[op1]], {{.*}}truncate-all.cpp:[[op1loc]]
// CHECK-LOCS-NEXT: 0x[[op2]], {{.*}}truncate-all.cpp:[[op2loc]]
// CHECK-LOCS-NEXT: 0x[[op3]], {{.*}}truncate-all.cpp:[[op3loc]]
// CHECK-LOCS-NEXT: 0x[[op4]], {{.*}}truncate-all-header.h:[[op4loc]]
// CHECK-LOCS-NEXT: 0x[[op5]], {{.*}}truncate-all-header.h:[[op5loc]]
// CHECK-LOCS-NEXT: 0x[[op6]], {{.*}}truncate-all-header.h:[[op6loc]]
// CHECK-LOCS-NEXT: 0x[[op7]], {{.*}}truncate-all.cpp:[[op7loc]]


#include "truncate-all-header.h"

#include "../test_utils.h"

__attribute__((noinline))
floatty simple_add(floatty a, floatty b) {
    return a + b;
}
__attribute__((noinline))
floatty intrinsics(floatty a, floatty b) {
    return sqrt(a) * pow(b, 2);
}
__attribute__((noinline))
floatty compute2(floatty *A, floatty *B, floatty *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] / 2 + intrinsics2(A[i], simple_add(B[i] * 10000, 0.000001));
    }
    return C[0];
}
__attribute__((noinline))
floatty compute(floatty *A, floatty *B, floatty *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] / 2 + intrinsics(A[i], simple_add(B[i] * 10000, 0.000001));
    }
    return C[0];
}

int main() {
    floatty A[N];
    floatty B[N];
    floatty C[N];

    for (int i = 0; i < N; i++) {
        A[i] = 1 + i % 5;
        B[i] = 1 + i % 3;
    }

    compute2(A, B, C, N);
    for (int i = 0; i < N; i++)
        C[i] = 0;
    compute(A, B, C, N);
    printf("%f\n", C[5]);
}
