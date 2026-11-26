// The driver's translation: the plugins (Enzyme's after Poseidon's to generate
// a profile), the include directory, -ffp-contract=on, POSEIDON_DEVICE_MATH,
// each -poseidon-<flag> through -Xclang, and the runtimes the action needs.
// REQUIRES: enzyme
// DEFINE: %{echo} = env POSEIDON_ECHO=1 %poseidon_clangxx
// DEFINE: %{cuda} = -x cuda --cuda-gpu-arch=%gpu_arch --target=x86_64-unknown-linux-gnu --cuda-path=%t.cuda -nocudainc -nocudalib
// DEFINE: %{hip} = -x hip --offload-arch=gfx942 --target=x86_64-unknown-linux-gnu --rocm-path=%t.rocm -nogpuinc -nogpulib
// RUN: rm -rf %t && mkdir -p %t/tmp && echo > %t.link.o && cp %s %t.cxx.cu
// -x c++ takes no CUDA runtime; a flag in the driver's own form still asks for
// the profiler; a C link takes the C++ library; objects alone take the GPU
// runtimes as needed.
// RUN: %{echo} -poseidon-profile-use -x c++ %t.cxx.cu -### -o %t.exe 2>&1 | FileCheck --check-prefix=XCXX %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clang -Xclang -mllvm -Xclang -poseidon-profile-generate -x c %s -### -o %t.exe 2>&1 | FileCheck --check-prefix=XCLANG %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clang -poseidon-profile-use %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=CLINK %s
// RUN: env POSEIDON_ECHO=1 %poseidon_clang -poseidon-profile-use -stdlib=libc++ %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=CLINKCXX %s
// RUN: %if cuda-toolkit %{ %{echo} -poseidon-profile-use %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=OBJCUDA %s %}
// RUN: %{echo} -poseidon-profile-use -poseidon-tau=1e-7 -poseidon-cache=%t.cache %{cuda} -### %s -o %t.o 2>&1 | FileCheck --check-prefix=USE --implicit-check-not=ClangEnzyme %s
// RUN: %{echo} -poseidon-profile-generate %{cuda} -### %s -o %t.o 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: %poseidon_clangxx -poseidon-help | FileCheck --check-prefix=HELP %s
// The GPU language from a joined -x and from an AMD --cuda-gpu-arch; a line
// that only links is not warned about the flags it carries.
// RUN: env POSEIDON_ECHO=1 POSEIDON_ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so %poseidon_clangxx -xhip --cuda-gpu-arch=gfx90a --target=x86_64-unknown-linux-gnu --rocm-path=%t.rocm -nogpuinc -nogpulib -poseidon-profile-generate -poseidon-tau=1e-12 -c -### %s -o %t.o 2>&1 | FileCheck --check-prefix=XHIP %s
// RUN: %{echo} -Werror -poseidon-profile-use -poseidon-tau=1e-7 %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=LINKONLY %s
// A separate value of a clang option stays, also one that spells a Poseidon
// flag or a CUDA source.
// RUN: %{echo} -MT -poseidon-tau=1 -MD -MF %t.dep.cu -I -poseidon-include %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=VALUES %s
// The runtimes go to a line that links: not to one that writes dependencies, a
// module or an AST, nor to one without an input.
// RUN: %poseidon_clangxx -Werror -poseidon-profile-generate -MM -MT %t.o -x c %s | FileCheck --check-prefix=DEPS %s
// RUN: %{echo} -Werror -poseidon-profile-use -emit-ast -x c %s -### -o %t.ast 2>&1 | FileCheck --check-prefix=NOLINK %s
// RUN: %{echo} -Werror -poseidon-profile-use --precompile -x c++-header %s -### -o %t.pcm 2>&1 | FileCheck --check-prefix=NOLINK %s
// RUN: %{echo} -v 2>&1 | FileCheck --check-prefix=VERSION %s
// A value pair reaches clang adjacent and unrewritten; a Poseidon flag after
// -mllvm is spelled as any other.
// RUN: %{echo} -poseidon-profile-use=%t.profile -Xcuda-ptxas --maxrregcount=160 -Xclang -poseidon-not-a-flag -mllvm -poseidon-print -mllvm=-poseidon-enable-rewrite=0 %{cuda} -### %s -o %t.o 2>&1 | FileCheck --check-prefix=XPAIR %s
// HIP: the flags reach the compilers, not the device linker; the runtimes are
// HIP's, from HIP_PATH before --rocm-path, ROCM_PATH without either.
// RUN: %{echo} -poseidon-profile-use -poseidon-tau=1e-7 %{hip} -### %s -o %t.o 2>&1 | FileCheck --check-prefix=HIPUSE --implicit-check-not=ClangEnzyme --implicit-check-not=enzyme-lto-prelink %s
// RUN: %{echo} -poseidon-profile-generate %{hip} -### %s -o %t.o 2>&1 | FileCheck --check-prefix=HIPGEN %s
// RUN: mkdir -p %t.hippath && env HIP_PATH=%t.hippath %{echo} -poseidon-profile-use %{hip} -### %s -o %t.o 2>&1 | FileCheck --check-prefix=HIPPATH %s
// RUN: mkdir -p %t.rocmpath && env ROCM_PATH=%t.rocmpath %{echo} -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc --target=x86_64-unknown-linux-gnu -nogpulib -### %s -o %t.o 2>&1 | FileCheck --check-prefix=ROCMPATH %s
// What clang prints for -###, -v, --version and -print-* comes once.
// RUN: %poseidon_clang -### -c %s -o %t.o 2>&1 | grep -c 'clang version' | FileCheck --check-prefix=ONCE %s
// RUN: %poseidon_clang --version | grep -c 'clang version' | FileCheck --check-prefix=ONCE %s
// RUN: %poseidon_clang -print-resource-dir | count 1
// Arguments after response files are expanded: nested flags reach the pass, a
// file of flags alone is no input, and a command too long for the system goes
// to clang in a response file the driver removes.
// RUN: printf -- '-poseidon-profile-use -poseidon-tau=1e-7\n' > %t/flags.rsp
// RUN: printf -- '@%t/flags.rsp -poseidon-cache=%t/cache\n' > %t/nested.rsp
// RUN: %{echo} @%t/nested.rsp -x c -### %s -o %t/a.out 2>&1 | FileCheck --check-prefix=RSP %s
// RUN: printf -- '-v\n' > %t/version.rsp
// RUN: %{echo} @%t/version.rsp 2>&1 | FileCheck --check-prefix=RSP-VERSION %s
// RUN: %python -c "print('-DLONG=' + 'a' * 140000)" > %t/long.rsp
// RUN: env TMPDIR=%t/tmp %poseidon_clang -Werror -fsyntax-only -DEXPECT_LONG -x c @%t/long.rsp %s
// RUN: ls %t/tmp | count 0

#if defined(EXPECT_LONG) && !defined(LONG)
#error "the macro of the response file did not arrive"
#endif
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

// RSP: clang{{.*}} -fpass-plugin={{.*}}Poseidon-{{[0-9]+}}.so
// RSP-SAME: -Xclang -mllvm -Xclang -poseidon-profile-use
// RSP-SAME: -Xclang -mllvm -Xclang -poseidon-tau=1e-7
// RSP-SAME: -Xclang -mllvm -Xclang -poseidon-cache=
// RSP-NOT: unknown argument

// RSP-VERSION: clang++ -v{{ ?$}}
// RSP-VERSION-NEXT: clang version
// RSP-VERSION-NOT: "{{[^"]*}}ld{{(\.lld)?}}"
