// The driver's translation: two pass plugins in pipeline order, Poseidon's
// include directory, statement-only FP contraction, every -poseidon-<flag>
// forwarded to the pass through -Xclang, and the runtime the selected action
// needs. Naming
// the CUDA path and dropping its headers and libdevice keeps clang's exit
// status, and -lcudart, independent of the host's CUDA.
//
// REQUIRES: enzyme
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
// The GPU language from a joined -x and from the processor, also when
// --cuda-gpu-arch names an AMD one: a HIP compile gets the HIP Enzyme option,
// its compilers the Poseidon flags, and its device link none. A line that only
// links is not warned about the flags it carries.
// RUN: env POSEIDON_ECHO=1 ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so \
// RUN:     %poseidon_clangxx -xhip --cuda-gpu-arch=gfx90a --rocm-path=%t.rocm \
// RUN:     -nogpuinc -nogpulib -poseidon-tau=1e-12 -c -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=XHIP %s
// RUN: echo > %t.link.o
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -Werror -poseidon-profile-use \
// RUN:     -poseidon-tau=1e-7 %t.link.o -### -o %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=LINKONLY %s
//
// Every option of clang's whose value is the next argv entry keeps it, also
// where the value spells a Poseidon flag or a CUDA source.
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -MT -poseidon-tau=1 -MD \
// RUN:     -MF %t.dep.cu -I -poseidon-include %t.link.o -### -o %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=VALUES %s
//
// The runtimes go to a line that links: not to one that only writes
// dependencies, a module or an AST, and not to one without an input, which
// gets the user's arguments alone.
// RUN: %poseidon_clangxx -Werror -poseidon-profile-generate -MM -MT %t.o -x c %s \
// RUN:   | FileCheck --check-prefix=DEPS %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -Werror -poseidon-profile-use \
// RUN:     -emit-ast -x c %s -### -o %t.ast 2>&1 \
// RUN:   | FileCheck --check-prefix=NOLINK %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -Werror -poseidon-profile-use \
// RUN:     --precompile -x c++-header %s -### -o %t.pcm 2>&1 \
// RUN:   | FileCheck --check-prefix=NOLINK %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -v 2>&1 \
// RUN:   | FileCheck --check-prefix=VERSION %s
//
// An option whose value is a separate argv entry keeps that value: the pair
// reaches clang adjacent, and a value that spells a Poseidon flag is NOT
// rewritten into a flag of its own. A Poseidon flag the user wrote after
// -mllvm is spelled as any other.
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use=%t.profile \
// RUN:     -Xcuda-ptxas --maxrregcount=160 -Xclang -poseidon-not-a-flag \
// RUN:     -mllvm -poseidon-print -mllvm=-poseidon-enable-herbie=0 \
// RUN:     -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=XPAIR %s
//
// HIP: every -poseidon-<flag> reaches the compilers and not the device
// linker, to which HIP's driver hands -mllvm, and the runtimes are HIP's.
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-tau=1e-7 -x hip --offload-arch=gfx942 \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPUSE %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-generate \
// RUN:     -x hip --offload-arch=gfx942 \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPGEN %s
//
// The HIP runtime's libraries come from where clang takes the HIP runtime:
// HIP_PATH before --rocm-path, ROCM_PATH without either.
// RUN: mkdir -p %t.hippath && env POSEIDON_ECHO=1 HIP_PATH=%t.hippath %poseidon_clangxx \
// RUN:     -poseidon-profile-use -x hip --offload-arch=gfx942 \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPPATH %s
// RUN: mkdir -p %t.rocmpath && env POSEIDON_ECHO=1 ROCM_PATH=%t.rocmpath %poseidon_clangxx \
// RUN:     -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc \
// RUN:     -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ROCMPATH %s

int main(void) { return 0; }

// USE: clang{{.*}} -fpass-plugin={{.*}}Poseidon-{{[0-9]+}}.so -fpass-plugin={{.*}}Enzyme-{{[0-9]+}}.so -isystem {{.*}}/include -ffp-contract=on
// USE-SAME: -Xclang -mllvm -Xclang -poseidon-profile-use -Xclang -mllvm -Xclang -poseidon-tau=1e-7 -Xclang -mllvm -Xclang -poseidon-cache=
// USE-SAME: -lposeidon_profile -lposeidon_rt -lcublas

// GEN: clang{{.*}} -fpass-plugin={{.*}}Poseidon-{{[0-9]+}}.so -fpass-plugin={{.*}}Enzyme-{{[0-9]+}}.so
// GEN-SAME: -Xclang -mllvm -Xclang -poseidon-profile-generate
// No relocatable device code and no runtime source: the user's arguments are
// followed by the runtimes alone.
// GEN-SAME: -### {{[^ ]+}} -o {{[^ ]+}} -L{{[^ ]+}} -lcudart -L{{[^ ]+}} -lposeidon_profile_host{{( -lposeidon_profile_cuda)?}} -lposeidon_profile{{ ?$}}

// XHIP: clang{{.*}} -Xclang -mllvm -Xclang -enzyme-lto-prelink=1
// XHIP-SAME: -xhip --cuda-gpu-arch=gfx90a
// XHIP-SAME: -Xclang -mllvm -Xclang -poseidon-tau=1e-12
// XHIP: "-cc1" {{.*}}"-fcuda-is-device"{{.*}} "-mllvm" "-poseidon-tau=1e-12"
// XHIP-NOT: {{[="]}}-poseidon-
// XHIP: clang-linker-wrapper"
// XHIP-NOT: {{[="]}}-poseidon-
// XHIP: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-tau=1e-12"

// LINKONLY: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-use -Xclang -mllvm -Xclang -poseidon-tau=1e-7
// LINKONLY-NOT: argument unused
// LINKONLY: "{{[^"]*}}ld{{(\.lld)?}}"
// LINKONLY-NOT: argument unused

// VALUES: clang{{.*}} -MT -poseidon-tau=1 -MD -MF {{[^ ]+}}.dep.cu -I -poseidon-include
// VALUES-NOT: -Xclang -mllvm -Xclang -poseidon-tau
// VALUES-NOT: -lcudart

// DEPS: {{.*}}.o: {{\\$}}
// DEPS-NEXT: {{.*}}driver_flags.c

// NOLINK: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-
// NOLINK-NOT: -lposeidon
// NOLINK-NOT: argument unused
// NOLINK-NOT: "{{[^"]*}}ld{{(\.lld)?}}"

// VERSION: clang++ -v{{ ?$}}
// VERSION-NEXT: clang version
// VERSION-NOT: argument unused
// VERSION-NOT: "{{[^"]*}}ld{{(\.lld)?}}"

// HELP: -poseidon-profile-generate
// HELP: -poseidon-profile-use
// HELP: -poseidon-tau
// HELP-NOT: -poseidon-confidence

// XPAIR: clang{{.*}} -Xcuda-ptxas --maxrregcount=160
// XPAIR-SAME: -Xclang -poseidon-not-a-flag
// XPAIR-SAME: -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-herbie=0
// XPAIR-NOT: -mllvm --maxrregcount=160
// XPAIR-NOT: -mllvm -poseidon-not-a-flag
// XPAIR-NOT: -mllvm{{=| }}-poseidon

// HIPUSE: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-use -Xclang -mllvm -Xclang -poseidon-tau=1e-7
// HIPUSE-NOT: -mllvm -poseidon-
// HIPUSE-SAME: -L{{[^ ]+}}.rocm/lib -lamdhip64
// HIPUSE-SAME: -lposeidon_profile -lposeidon_rt_hip -lhipblas{{ ?$}}

// HIPGEN: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-generate
// HIPGEN-SAME: -### {{[^ ]+}} -o {{[^ ]+}} -L{{[^ ]+}}.rocm/lib -lamdhip64 -L{{[^ ]+}} -lposeidon_profile_host{{( -lposeidon_profile_hip)?}} -lposeidon_profile{{ ?$}}

// HIPPATH: clang{{.*}} -L{{[^ ]+}}.hippath/lib -lamdhip64

// ROCMPATH: clang{{.*}} -L{{[^ ]+}}.rocmpath/lib -lamdhip64
