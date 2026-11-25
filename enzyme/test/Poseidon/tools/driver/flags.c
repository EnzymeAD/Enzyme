// The driver's translation: Poseidon's pass plugin, and Enzyme's after it to
// generate a profile, Poseidon's include directory, statement-only FP
// contraction, POSEIDON_DEVICE_MATH, every -poseidon-<flag> forwarded to the
// pass through -Xclang, and the runtime the selected action needs. Naming the
// CUDA path and dropping its headers and libdevice keeps clang's exit status,
// and -lcudart, independent of the host's CUDA.
//
// REQUIRES: enzyme
//
// The command line is read as clang reads it: -x c++ compiles a .cu file as
// C++, which takes no CUDA runtime; a flag written in the form the driver
// gives it still asks for the profiler; a C link takes the C++ library the
// Poseidon archives need; and a link of objects alone takes the build's GPU
// runtimes, recorded as needed only if used.
// RUN: echo > %t.link.o
// RUN: cp %s %t.cxx.cu
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use -x c++ \
// RUN:     %t.cxx.cu -### -o %t.exe 2>&1 | FileCheck --check-prefix=XCXX %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clang -Xclang -mllvm -Xclang \
// RUN:     -poseidon-profile-generate -x c %s -### -o %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=XCLANG %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clang -poseidon-profile-use %t.link.o \
// RUN:     -### -o %t.exe 2>&1 | FileCheck --check-prefix=CLINK %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clang -poseidon-profile-use -stdlib=libc++ \
// RUN:     %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=CLINKCXX %s
// RUN: %if cuda-toolkit %{ env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=OBJCUDA %s %}
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-tau=1e-7 -poseidon-cache=%t.cache \
// RUN:     -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=USE --implicit-check-not=ClangEnzyme %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-generate \
// RUN:     -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=GEN %s
// RUN: %poseidon_clangxx -poseidon-help | FileCheck --check-prefix=HELP %s
//
// The GPU language from a joined -x and from the processor, also when
// --cuda-gpu-arch names an AMD one: a HIP compile that loads Enzyme gets the
// HIP Enzyme option, its compilers the Poseidon flags, and its device link
// none. A line that only links is not warned about the flags it carries.
// RUN: env POSEIDON_ECHO=1 POSEIDON_ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so \
// RUN:     %poseidon_clangxx -xhip --cuda-gpu-arch=gfx90a \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --rocm-path=%t.rocm \
// RUN:     -nogpuinc -nogpulib -poseidon-profile-generate -poseidon-tau=1e-12 \
// RUN:     -c -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=XHIP %s
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
// RUN:     -mllvm -poseidon-print -mllvm=-poseidon-enable-rewrite=0 \
// RUN:     -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=XPAIR %s
//
// HIP: every -poseidon-<flag> reaches the compilers and not the device
// linker, to which HIP's driver hands -mllvm, and the runtimes are HIP's.
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-tau=1e-7 -x hip --offload-arch=gfx942 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPUSE --implicit-check-not=ClangEnzyme \
// RUN:     --implicit-check-not=enzyme-lto-prelink %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-generate \
// RUN:     -x hip --target=x86_64-unknown-linux-gnu --offload-arch=gfx942 \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPGEN %s
//
// The HIP runtime's libraries come from where clang takes the HIP runtime:
// HIP_PATH before --rocm-path, ROCM_PATH without either.
// RUN: mkdir -p %t.hippath && env POSEIDON_ECHO=1 HIP_PATH=%t.hippath %poseidon_clangxx \
// RUN:     -poseidon-profile-use -x hip --offload-arch=gfx942 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPPATH %s
// RUN: mkdir -p %t.rocmpath && env POSEIDON_ECHO=1 ROCM_PATH=%t.rocmpath %poseidon_clangxx \
// RUN:     -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -nogpulib -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ROCMPATH %s
// What clang prints for -###, -v, --version and -print-* comes once.
// RUN: %poseidon_clang -### -c %s -o %t.o 2>&1 | grep -c 'clang version' \
// RUN:   | FileCheck --check-prefix=ONCE %s
// RUN: %poseidon_clang --version | grep -c 'clang version' \
// RUN:   | FileCheck --check-prefix=ONCE %s
// RUN: %poseidon_clang -print-resource-dir | count 1

int main(void) { return 0; }

// USE: clang{{.*}} -fpass-plugin={{.*}}Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-isystem {{.*}}/include -ffp-contract=on -DPOSEIDON_DEVICE_MATH
// USE-SAME: -Xclang -mllvm -Xclang -poseidon-profile-use -Xclang -mllvm -Xclang -poseidon-tau=1e-7 -Xclang -mllvm -Xclang -poseidon-cache=
// USE-SAME: -### {{[^ ]+}} -o {{[^ ]+}} -L{{[^ ]+}}.cuda/lib64 -lcudart -L{{[^ ]+}} -lposeidon_profile -lposeidon_gemm_cuda -Wl,--push-state,--as-needed -lcublas -Wl,--pop-state{{( -Xlinker [^ ]+/libcumpsgemm.so -Xlinker -rpath -Xlinker [^ ]+)?}}{{ ?$}}
// USE: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}} "-fpass-plugin={{[^"]*}}Poseidon-{{[0-9]+}}.so"{{.*}} "-mllvm" "-poseidon-profile-use" "-mllvm" "-poseidon-tau=1e-7" "-mllvm" "-poseidon-cache={{[^"]+}}.cache"
// USE-NOT: {{[="]}}-poseidon-
// USE: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-fpass-plugin={{[^"]*}}Poseidon-{{[0-9]+}}.so"{{.*}} "-mllvm" "-poseidon-profile-use" "-mllvm" "-poseidon-tau=1e-7" "-mllvm" "-poseidon-cache={{[^"]+}}.cache"
// USE-NOT: {{[="]}}-poseidon-
// USE: "-lcudart" "-lposeidon_profile" "-lposeidon_gemm_cuda" "--push-state" "--as-needed" "-lcublas" "--pop-state"

// GEN: clang{{.*}} -fpass-plugin={{.*}}Poseidon-{{[0-9]+}}.so -fpass-plugin={{.*}}Enzyme-{{[0-9]+}}.so
// GEN-SAME: -Xclang -mllvm -Xclang -poseidon-profile-generate
// No relocatable device code and no runtime source: the user's arguments are
// followed by the runtimes alone.
// GEN-SAME: -### {{[^ ]+}} -o {{[^ ]+}} -L{{[^ ]+}}.cuda/lib64 -lcudart -L{{[^ ]+}} -lposeidon_profile_shadows{{( -lposeidon_profile_cuda)?}} -lposeidon_profile{{ ?$}}
// GEN: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}} "-fpass-plugin={{[^"]*}}Poseidon-{{[0-9]+}}.so" "-fpass-plugin={{[^"]*}}Enzyme-{{[0-9]+}}.so"{{.*}} "-mllvm" "-poseidon-profile-generate"
// GEN-NOT: {{[="]}}-poseidon-
// GEN: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-fpass-plugin={{[^"]*}}Poseidon-{{[0-9]+}}.so" "-fpass-plugin={{[^"]*}}Enzyme-{{[0-9]+}}.so"{{.*}} "-mllvm" "-poseidon-profile-generate"
// GEN-NOT: {{[="]}}-poseidon-
// GEN: "-lcudart" "-lposeidon_profile_shadows"{{( "-lposeidon_profile_cuda")?}} "-lposeidon_profile"

// XHIP: clang{{.*}} -Xclang -mllvm -Xclang -enzyme-lto-prelink=1
// XHIP-SAME: -xhip --cuda-gpu-arch=gfx90a
// XHIP-SAME: -Xclang -mllvm -Xclang -poseidon-tau=1e-12
// XHIP: "-cc1" {{.*}}"-fcuda-is-device"{{.*}} "-mllvm" "-poseidon-tau=1e-12"
// XHIP-NOT: {{[="]}}-poseidon-
// XHIP: {{(clang-linker-wrapper|[/"]lld)}}"
// XHIP-NOT: {{[="]}}-poseidon-
// XHIP: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-tau=1e-12"

// LINKONLY: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-use -Xclang -mllvm -Xclang -poseidon-tau=1e-7
// LINKONLY-NOT: argument unused
// LINKONLY: "{{[^"]*}}ld{{(\.lld)?}}"
// LINKONLY-NOT: argument unused

// VALUES: clang{{.*}} -MT -poseidon-tau=1 -MD -MF {{[^ ]+}}.dep.cu -I -poseidon-include
// VALUES-NOT: -Xclang -mllvm -Xclang -poseidon-tau
// VALUES-NOT: lib64 -lcudart

// DEPS: {{.*}}.o: {{\\$}}
// DEPS-NEXT: {{.*}}flags.c

// NOLINK: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-
// NOLINK-NOT: -lposeidon
// NOLINK-NOT: argument unused
// NOLINK-NOT: "{{[^"]*}}ld{{(\.lld)?}}"

// VERSION: clang++ -v{{ ?$}}
// VERSION-NEXT: clang version
// VERSION-NOT: argument unused
// VERSION-NOT: "{{[^"]*}}ld{{(\.lld)?}}"

// HELP: 1. compile with -poseidon-profile-generate
// HELP: POSEIDON_PROFILE_DIR (default ./poseidon.profile)
// HELP: poseidon-kappa -- <program>
// HELP: in 1 + 3 x <sites> more runs
// HELP: 3. compile with -poseidon-profile-use
// HELP: Profiling a host function needs glibc
// HELP: -poseidon-profile-generate
// HELP: -poseidon-profile-use
// HELP: -poseidon-tau
// HELP-NOT: -poseidon-confidence

// XPAIR: clang{{.*}} -Xcuda-ptxas --maxrregcount=160
// XPAIR-SAME: -Xclang -poseidon-not-a-flag
// XPAIR-SAME: -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0
// XPAIR-NOT: -mllvm --maxrregcount=160
// XPAIR-NOT: -mllvm -poseidon-not-a-flag
// XPAIR-NOT: -mllvm{{=| }}-poseidon

// HIPUSE: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-profile-use -Xclang -mllvm -Xclang -poseidon-tau=1e-7
// HIPUSE-NOT: -mllvm -poseidon-
// HIPUSE-SAME: -L{{[^ ]+}}.rocm/lib -lamdhip64
// HIPUSE-SAME: -lposeidon_profile -lposeidon_gemm_hip -Wl,--push-state,--as-needed -lhipblas -Wl,--pop-state{{ ?$}}
// HIPUSE: "-cc1" {{.*}}"-fcuda-is-device"{{.*}} "-fpass-plugin={{[^"]*}}Poseidon-{{[0-9]+}}.so"{{.*}} "-mllvm" "-poseidon-profile-use" "-mllvm" "-poseidon-tau=1e-7"
// HIPUSE-NOT: {{[="]}}-poseidon-
// HIPUSE: {{(clang-linker-wrapper|[/"]lld)}}"
// HIPUSE-NOT: {{[="]}}-poseidon-
// HIPUSE: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-fpass-plugin={{[^"]*}}Poseidon-{{[0-9]+}}.so"{{.*}} "-mllvm" "-poseidon-profile-use" "-mllvm" "-poseidon-tau=1e-7"
// HIPUSE-NOT: {{[="]}}-poseidon-
// HIPUSE: "-lamdhip64" "-lposeidon_profile" "-lposeidon_gemm_hip" "--push-state" "--as-needed" "-lhipblas" "--pop-state"

// HIPGEN: clang{{.*}} -fpass-plugin={{[^ ]*}}Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-fpass-plugin={{[^ ]*}}Enzyme-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-Xclang -mllvm -Xclang -enzyme-lto-prelink=1 -isystem {{[^ ]*}}/include -ffp-contract=on -DPOSEIDON_DEVICE_MATH
// HIPGEN-SAME: -Xclang -mllvm -Xclang -poseidon-profile-generate
// HIPGEN-SAME: -### {{[^ ]+}} -o {{[^ ]+}} -L{{[^ ]+}}.rocm/lib -lamdhip64 -L{{[^ ]+}} -lposeidon_profile_shadows{{( -lposeidon_profile_hip)?}} -lposeidon_profile{{ ?$}}
// HIPGEN: "-cc1" {{.*}}"-fcuda-is-device"{{.*}} "-fpass-plugin={{[^"]*}}Poseidon-{{[0-9]+}}.so" "-fpass-plugin={{[^"]*}}Enzyme-{{[0-9]+}}.so"{{.*}} "-mllvm" "-enzyme-lto-prelink=1"{{.*}} "-mllvm" "-poseidon-profile-generate"
// HIPGEN-NOT: {{[="]}}-poseidon-
// HIPGEN: {{(clang-linker-wrapper|[/"]lld)}}"
// HIPGEN-NOT: {{[="]}}-poseidon-
// HIPGEN: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-fpass-plugin={{[^"]*}}Poseidon-{{[0-9]+}}.so" "-fpass-plugin={{[^"]*}}Enzyme-{{[0-9]+}}.so"{{.*}} "-mllvm" "-enzyme-lto-prelink=1"{{.*}} "-mllvm" "-poseidon-profile-generate"
// HIPGEN-NOT: {{[="]}}-poseidon-
// HIPGEN: "-lamdhip64" "-lposeidon_profile_shadows"{{( "-lposeidon_profile_hip")?}} "-lposeidon_profile"

// HIPPATH: clang{{.*}} -L{{[^ ]+}}.hippath/lib -lamdhip64

// ROCMPATH: clang{{.*}} -L{{[^ ]+}}.rocmpath/lib -lamdhip64

// XCXX: clang++ {{.*}}.cxx.cu -### -o {{[^ ]+}} -L{{[^ ]+}} -lposeidon_profile{{ ?$}}

// XCLANG: clang {{.*}}-fpass-plugin={{.*}}Enzyme-{{[0-9]+}}.so
// XCLANG-SAME: -Xclang -mllvm -Xclang -poseidon-profile-generate
// XCLANG-SAME: -lposeidon_profile_shadows{{.*}} -lposeidon_profile {{.*}}-lstdc++ -lm{{ ?$}}

// CLINK: clang {{.*}}.link.o -### -o {{[^ ]+}} {{.*}}-lposeidon_profile {{.*}}-lstdc++ -lm{{ ?$}}
// CLINKCXX: clang {{.*}}-lposeidon_profile {{.*}}-lc++ -lm{{ ?$}}

// OBJCUDA: clang++ {{.*}}.link.o -### -o {{[^ ]+}} -L{{[^ ]+}}/lib64 -Wl,--push-state,--as-needed -lcudart -Wl,--pop-state -L{{[^ ]+}} -lposeidon_profile -lposeidon_gemm_cuda -Wl,--push-state,--as-needed -lcublas -Wl,--pop-state

// ONCE: {{^}}1{{$}}
