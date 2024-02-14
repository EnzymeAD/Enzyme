// Baseline
// RUN: export ENZYME_TRUNCATE_ALL=""; if [ %llvmver -ge 12 ]; then [ "$(%clang -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -S | %lli -)" == "900000000.560000" ] ; fi

// Truncated
// RUN: export ENZYME_TRUNCATE_ALL="64to32"; if [ %llvmver -ge 12 ]; then [ "$(%clang -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -S | %lli -)" == "900000000.000000" ] ; fi
// RUN: export ENZYME_TRUNCATE_ALL="11-52to8-23"; if [ %llvmver -ge 12 ]; then [ "$(%clang -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -S | %lli -)" == "900000000.000000" ] ; fi

#include <math.h>

#include "../test_utils.h"

#define N 10

#define floatty double


__attribute__((noinline))
floatty simple_add(floatty a, floatty b) {
    return a + b;
}
__attribute__((noinline))
floatty intrinsics(floatty a, floatty b) {
    return sqrt(a) * pow(b, 2);
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

    compute(A, B, C, N);
    printf("%f\n", C[5]);
}
