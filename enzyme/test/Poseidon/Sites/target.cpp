// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x c++ -std=c++17 -O1 -S -emit-llvm -I%poseidon_profile_inc %s -o %t/plain.ll
// RUN: FileCheck --check-prefix=PLAIN %s < %t/plain.ll
// RUN: %clang -x c++ -std=c++17 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-print %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clang -x c++ -std=c++17 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -DBAD=0 %s -o /dev/null 2>&1 | FileCheck --check-prefix=ZERO %s
// RUN: not %clang -x c++ -std=c++17 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile "-DBAD=-1e-7" %s -o /dev/null 2>&1 | FileCheck --check-prefix=NEG %s
// RUN: not %clang -x c++ -std=c++17 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile "-DBAD=__builtin_inf()" %s -o /dev/null 2>&1 | FileCheck --check-prefix=INF %s
// RUN: not %clang -x c++ -std=c++17 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -DTWO %s -o /dev/null 2>&1 | FileCheck --check-prefix=TWO %s
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: %if cuda-toolkit %{ %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-print %s -o %t/k.ll 2>&1 | FileCheck --check-prefix=CUDA %s %}

// POSEIDON_OPTIMIZE_TAU(t) takes a literal, a macro or a constant expression;
// a target that is not a positive finite number, or other arguments, is an
// error naming the function. Without the plugin the source compiles. The same
// on a kernel, in the device compilation of a CUDA source (CUDA).

#include "poseidon/poseidon.h"

#define MY_TOL 1e-7
constexpr double kTol = 2.5e-9;

POSEIDON_OPTIMIZE_TAU(1e-6) double lit(const double *x) { return x[0] * x[0] + x[1]; }
POSEIDON_OPTIMIZE_TAU(MY_TOL) double mac(const double *x) { return x[0] * x[1] + x[0]; }
POSEIDON_OPTIMIZE_TAU(kTol) double cexpr(const double *x) { return x[1] * x[1] - x[0]; }
POSEIDON_OPTIMIZE double bare(const double *x) { return x[1] * x[0] - x[1]; }

struct Body {
  POSEIDON_OPTIMIZE_TAU(kTol * 2) double member(const double *x);
};
double Body::member(const double *x) { return x[0] * x[0] - x[1]; }

template <typename T> POSEIDON_OPTIMIZE_TAU(4e-4) T tmpl(const T *x) { return x[0] * x[1] - x[0]; }
template float tmpl<float>(const float *);

#ifdef __CUDA__
POSEIDON_OPTIMIZE_TAU(1e-6) __attribute__((global)) void lit(double *x) { x[0] = x[0] * x[0] + x[1]; }
POSEIDON_OPTIMIZE_TAU(MY_TOL) __attribute__((global)) void mac(double *x) { x[0] = x[0] * x[1] + x[1]; }
#endif
#ifdef BAD
POSEIDON_OPTIMIZE_TAU(BAD) double bad(const double *x) { return x[0] * x[0]; }
#endif
#ifdef TWO
__attribute__((annotate("poseidon", 1e-7, 2.0))) double two(const double *x) { return x[0] * x[0]; }
#endif

// PLAIN: @llvm.global.annotations
// PLAIN: define {{.*}}double @_Z3litPKd(

// CHECK: [poseidon] site lit(double const*): accuracy target 1e-06
// CHECK: [poseidon] site mac(double const*): accuracy target 1e-07
// CHECK: [poseidon] site cexpr(double const*): accuracy target 2.5e-09
// CHECK-NOT: site bare(double const*): accuracy target
// CHECK: [poseidon] site Body::member(double const*): accuracy target 5e-09
// CHECK: [poseidon] site float tmpl<float>(float const*): accuracy target 0.0004

// CUDA: [poseidon] site lit(double*): accuracy target 1e-06
// CUDA: [poseidon] site mac(double*): accuracy target 1e-07

// ZERO: error: {{.*}}Poseidon: bad(double const*) carries the accuracy target 0, which is not a positive finite number
// NEG: error: {{.*}}Poseidon: bad(double const*) carries the accuracy target -1e-07, which is not a positive finite number
// INF: error: {{.*}}Poseidon: bad(double const*) carries the accuracy target inf, which is not a positive finite number
// TWO: error: {{.*}}Poseidon: two(double const*) carries a Poseidon annotation whose arguments are not one accuracy target
