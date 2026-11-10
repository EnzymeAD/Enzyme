// The driver's translation: two pass plugins in pipeline order, Poseidon's
// include directory, statement-only FP contraction, every -poseidon-<flag>
// forwarded to the pass, and the runtime the selected action needs. Naming
// the CUDA path and dropping its headers and libdevice keeps clang's exit
// status, and -lcudart, independent of the host's CUDA.
//
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-tau=1e-7 -poseidon-cache=%t.cache \
// RUN:     -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=USE %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-generate \
// RUN:     -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=GEN %s
// RUN: %poseidon_clangxx -poseidon-help | FileCheck --check-prefix=HELP %s
//
// An option whose value is a separate argv entry keeps that value: the pair
// reaches clang adjacent, and a value that spells a Poseidon flag is NOT
// rewritten into an -mllvm flag of its own.
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use=%t.profile \
// RUN:     -Xcuda-ptxas --maxrregcount=160 -Xclang -poseidon-not-a-flag \
// RUN:     -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=XPAIR %s
//
// HIP: every -poseidon-<flag> reaches the compiler through -Xclang (HIP's
// driver hands -mllvm to the device linker as well), and the runtimes are
// HIP's.
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-tau=1e-7 -x hip --offload-arch=gfx942 \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPUSE %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-generate \
// RUN:     -x hip --offload-arch=gfx942 \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPGEN %s
//
// REQUIRES: poseidon, enzyme

int main(void) { return 0; }

// USE: clang{{.*}} -fpass-plugin={{.*}}Poseidon-{{[0-9]+}}.so -fpass-plugin={{.*}}Enzyme-{{[0-9]+}}.so -I{{.*}}/include -ffp-contract=on
// USE-SAME: -mllvm -poseidon-profile-use -mllvm -poseidon-tau=1e-7 -mllvm -poseidon-cache=
// USE-SAME: -lposeidon_profile -lposeidon_rt -lcublas

// GEN: clang{{.*}} -fpass-plugin={{.*}}Poseidon-{{[0-9]+}}.so -fpass-plugin={{.*}}Enzyme-{{[0-9]+}}.so
// GEN-SAME: -mllvm -poseidon-profile-generate
// No relocatable device code and no runtime source: the user's arguments are
// followed by the runtimes alone.
// GEN-SAME: -### {{[^ ]+}} -o {{[^ ]+}} -L{{[^ ]+}} -lcudart -L{{[^ ]+}} -lposeidon_profile_host{{( -lposeidon_profile_cuda)?}} -lposeidon_profile{{ ?$}}

// HELP: -poseidon-profile-generate
// HELP: -poseidon-profile-use
// HELP: -poseidon-tau
// HELP-NOT: -poseidon-confidence

// XPAIR: clang{{.*}} -Xcuda-ptxas --maxrregcount=160
// XPAIR-SAME: -Xclang -poseidon-not-a-flag
// XPAIR-NOT: -mllvm --maxrregcount=160
// XPAIR-NOT: -mllvm -poseidon-not-a-flag

// HIPUSE: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-use -Xclang -mllvm -Xclang -poseidon-tau=1e-7
// HIPUSE-NOT: -mllvm -poseidon-
// HIPUSE-SAME: -L{{[^ ]+}}.rocm/lib -lamdhip64
// HIPUSE-SAME: -lposeidon_profile -lposeidon_rt_hip -lhipblas{{ ?$}}

// HIPGEN: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-generate
// HIPGEN-SAME: -### {{[^ ]+}} -o {{[^ ]+}} -L{{[^ ]+}}.rocm/lib -lamdhip64 -L{{[^ ]+}} -lposeidon_profile_host{{( -lposeidon_profile_hip)?}} -lposeidon_profile{{ ?$}}
