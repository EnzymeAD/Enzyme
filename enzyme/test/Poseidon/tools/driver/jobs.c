// Which jobs get the driver's flags: one -poseidon-compilation-id per
// invocation (the same again for the same command, another for another, the
// user's own instead, also for one half), -enzyme-lto-prelink=1 for both
// compile jobs of a HIP compile that generates a profile; and the mistakes the
// driver refuses, with exit status 1.
// DEFINE: %{echo} = env POSEIDON_ECHO=1 %poseidon_clangxx
// DEFINE: %{cuda} = -x cuda --cuda-gpu-arch=%gpu_arch --target=x86_64-unknown-linux-gnu --cuda-path=%t.cuda -nocudainc -nocudalib
// DEFINE: %{hip} = -x hip --offload-arch=gfx942 --target=x86_64-unknown-linux-gnu --rocm-path=%t.rocm -nogpuinc -nogpulib
// RUN: %{echo} -poseidon-profile-use -poseidon-cache=%t.cache %{cuda} -### %s -o %t.o > %t.cuda.jobs 2>&1
// RUN: FileCheck --check-prefix=CUDA %s < %t.cuda.jobs
// RUN: grep -o 'poseidon-compilation-id=[^" ]*' %t.cuda.jobs > %t.cuda.units
// RUN: count 3 < %t.cuda.units
// RUN: sort -u %t.cuda.units | count 1
// RUN: %{echo} -poseidon-profile-use -poseidon-cache=%t.cache %{cuda} -### %s -o %t.o 2>&1 | grep -o 'poseidon-compilation-id=[^" ]*' | diff %t.cuda.units -
// RUN: %{echo} -poseidon-profile-use -poseidon-cache=%t.cache %{cuda} -### %s -o %t.other.o 2>&1 | grep -o 'poseidon-compilation-id=[^" ]*' > %t.other.units
// RUN: count 3 < %t.other.units
// RUN: not diff %t.cuda.units %t.other.units
// RUN: %{echo} -poseidon-profile-use %{hip} -### %s -o %t.o > %t.hip.jobs 2>&1
// RUN: FileCheck --check-prefix=HIP %s < %t.hip.jobs
// RUN: grep -o 'poseidon-compilation-id=[^" ]*' %t.hip.jobs > %t.hip.units
// RUN: count 3 < %t.hip.units
// RUN: sort -u %t.hip.units | count 1
// RUN: %{echo} -poseidon-profile-use -poseidon-compilation-id=mine %{cuda} -### %s -o %t.o > %t.mine.jobs 2>&1
// RUN: FileCheck --check-prefix=MINE %s < %t.mine.jobs
// RUN: FileCheck --check-prefix=MINE-UNITS --implicit-check-not=poseidon-compilation-id= %s < %t.mine.jobs
// RUN: %{echo} -poseidon-profile-use -Xarch_host -mllvm=-poseidon-compilation-id=mine %{hip} -### %s -o %t.o > %t.host.jobs 2>&1
// RUN: FileCheck --check-prefix=HOST %s < %t.host.jobs
// RUN: FileCheck --check-prefix=HOST-UNITS --implicit-check-not=poseidon-compilation-id= %s < %t.host.jobs
// RUN: echo > %t.link.o
// RUN: %{echo} -Werror %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=LINK %s
// The Enzyme plugin is only named, never loaded.
// RUN: env POSEIDON_ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so %{echo} -poseidon-profile-generate %{hip} -### %s -o %t.o 2>&1 | FileCheck --check-prefix=HIPLTO %s
// RUN: env POSEIDON_ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so %{echo} -Werror -poseidon-profile-generate --hip-link --rocm-path=%t.rocm %t.link.o -### -o %t.exe 2>&1 | FileCheck --check-prefix=HIPLINK %s
// RUN: env POSEIDON_ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so %{echo} -poseidon-profile-generate %{cuda} -### %s -o %t.o 2>&1 | FileCheck --check-prefix=CUDALTO --implicit-check-not=enzyme-lto-prelink %s
// Both halves of the workflow at once; a ROCm variable that names no directory.
// RUN: (%poseidon_clangxx -poseidon-profile-generate -poseidon-profile-use -c %s -o %t.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=MODES %s
// RUN: (env -u HIP_PATH ROCM_PATH=%t.none %poseidon_clangxx -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc -nogpulib -### %s -o %t.exe 2>&1; echo "exit status $?") | FileCheck --check-prefix=ROCM -DDIR=%t.none %s
// RUN: (env HIP_PATH=%t.none %poseidon_clangxx -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc -nogpulib -### %s -o %t.exe 2>&1; echo "exit status $?") | FileCheck --check-prefix=HIPDIR -DDIR=%t.none %s

int main(void) { return 0; }

// CUDA: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-compilation-id=[[UNIT:[0-9A-F]+]] -Xclang -mllvm -Xclang -poseidon-profile-use
// CUDA: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}} "-mllvm" "-poseidon-compilation-id=[[UNIT]]"
// CUDA: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-compilation-id=[[UNIT]]"
// CUDA-NOT: poseidon-compilation-id

// HIP: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-compilation-id=[[UNIT:[0-9A-F]+]] -Xclang -mllvm -Xclang -poseidon-profile-use
// HIP: "-cc1" {{.*}}"-fcuda-is-device"{{.*}} "-mllvm" "-poseidon-compilation-id=[[UNIT]]"
// HIP: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-compilation-id=[[UNIT]]"
// HIP-NOT: poseidon-compilation-id

// MINE: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-compilation-id=mine
// MINE: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}} "-mllvm" "-poseidon-compilation-id=mine"
// MINE: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-compilation-id=mine"
// MINE-NOT: poseidon-compilation-id
// MINE-UNITS-COUNT-3: poseidon-compilation-id=

// HOST: clang{{.*}} -Xarch_host -mllvm=-poseidon-compilation-id=mine
// HOST-NOT: poseidon-compilation-id
// HOST: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-compilation-id=mine"
// HOST-NOT: poseidon-compilation-id
// HOST-UNITS-COUNT-2: poseidon-compilation-id=

// LINK: clang{{.*}} -Xclang -mllvm -Xclang -poseidon-compilation-id={{[0-9A-F]+}}
// LINK-NOT: -poseidon-compilation-id=
// LINK-NOT: argument unused

// HIPLTO: clang{{.*}} -fpass-plugin={{[^ ]*}}/ClangEnzyme.so {{(-Xclang -load -Xclang [^ ]+ )?}}-Xclang -mllvm -Xclang -enzyme-lto-prelink=1 -isystem
// HIPLTO: "-cc1" {{.*}}"-fcuda-is-device"{{.*}} "-mllvm" "-enzyme-lto-prelink=1"
// HIPLTO-NOT: enzyme-lto-prelink
// HIPLTO: {{(clang-linker-wrapper|[/"]lld)}}"
// HIPLTO-NOT: enzyme-lto-prelink
// HIPLTO: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-enzyme-lto-prelink=1"
// HIPLTO-NOT: enzyme-lto-prelink

// HIPLINK: clang{{.*}} -Xclang -enzyme-lto-prelink=1
// HIPLINK-NOT: argument unused

// CUDALTO: clang{{.*}} -fpass-plugin={{[^ ]*}}/ClangEnzyme.so
// CUDALTO: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDALTO: "-cc1" "-triple" "x86_64-{{[^"]+}}"

// MODES: poseidon-clang++: error: Poseidon: -poseidon-profile-generate and -poseidon-profile-use are the two halves of the workflow; pass one{{$}}
// MODES-NEXT: exit status 1
// ROCM: poseidon-clang++: error: Poseidon: ROCM_PATH=[[DIR]] names no directory; set it to the ROCm installation or unset it{{$}}
// ROCM-NEXT: exit status 1
// HIPDIR: poseidon-clang++: error: Poseidon: HIP_PATH=[[DIR]] names no directory; set it to the ROCm installation or unset it{{$}}
// HIPDIR-NEXT: exit status 1
