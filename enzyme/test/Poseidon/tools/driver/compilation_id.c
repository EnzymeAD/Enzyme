// The driver names the translation unit of each invocation: one
// -poseidon-compilation-id reaches both compile jobs of a CUDA and of a HIP compile, the
// way every other Poseidon flag does, and no other job. The name follows from
// the working directory and the arguments, so the same invocation gets it
// again and another one does not. A unit the user names, also for one half
// only, is the only one. An invocation that only links passes the name to no
// job and is not warned about it.
//
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-cache=%t.cache -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o > %t.cuda.jobs 2>&1
// RUN: FileCheck --check-prefix=CUDA %s < %t.cuda.jobs
// RUN: grep -o 'poseidon-compilation-id=[^" ]*' %t.cuda.jobs > %t.cuda.units
// RUN: count 3 < %t.cuda.units
// RUN: sort -u %t.cuda.units | count 1
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-cache=%t.cache -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | grep -o 'poseidon-compilation-id=[^" ]*' | diff %t.cuda.units -
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-cache=%t.cache -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.other.o 2>&1 \
// RUN:   | grep -o 'poseidon-compilation-id=[^" ]*' > %t.other.units
// RUN: count 3 < %t.other.units
// RUN: not diff %t.cuda.units %t.other.units
//
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -x hip --offload-arch=gfx942 --rocm-path=%t.rocm -nogpuinc \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     -nogpulib -### %s -o %t.o > %t.hip.jobs 2>&1
// RUN: FileCheck --check-prefix=HIP %s < %t.hip.jobs
// RUN: grep -o 'poseidon-compilation-id=[^" ]*' %t.hip.jobs > %t.hip.units
// RUN: count 3 < %t.hip.units
// RUN: sort -u %t.hip.units | count 1
//
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-compilation-id=mine -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o > %t.mine.jobs 2>&1
// RUN: FileCheck --check-prefix=MINE %s < %t.mine.jobs
// RUN: FileCheck --check-prefix=MINE-UNITS --implicit-check-not=poseidon-compilation-id= %s < %t.mine.jobs
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -Xarch_host -mllvm=-poseidon-compilation-id=mine -x hip --offload-arch=gfx942 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o > %t.host.jobs 2>&1
// RUN: FileCheck --check-prefix=HOST %s < %t.host.jobs
// RUN: FileCheck --check-prefix=HOST-UNITS --implicit-check-not=poseidon-compilation-id= %s < %t.host.jobs
//
// RUN: echo > %t.link.o
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -Werror %t.link.o -### -o %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=LINK %s
//
// With Enzyme loaded to generate a profile, both compile jobs of a HIP compile
// get -enzyme-lto-prelink=1 (the device compile is an LTO pre-link compile
// whose device link loads no plugin); the device link does not, a HIP link is
// not warned about it, and a CUDA compile does not get it. The plugin is only
// named, never loaded.
// RUN: env POSEIDON_ECHO=1 POSEIDON_ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so \
// RUN:     %poseidon_clangxx -poseidon-profile-generate -x hip \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --offload-arch=gfx942 --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s \
// RUN:     -o %t.o 2>&1 | FileCheck --check-prefix=HIPLTO %s
// RUN: env POSEIDON_ECHO=1 POSEIDON_ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so \
// RUN:     %poseidon_clangxx -Werror -poseidon-profile-generate --hip-link \
// RUN:     --rocm-path=%t.rocm %t.link.o -### -o %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=HIPLINK %s
// RUN: env POSEIDON_ECHO=1 POSEIDON_ENZYME_PLUGIN=%t.enzyme/ClangEnzyme.so \
// RUN:     %poseidon_clangxx -poseidon-profile-generate -x cuda \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --cuda-gpu-arch=%gpu_arch --cuda-path=%t.cuda -nocudainc -nocudalib \
// RUN:     -### %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CUDALTO --implicit-check-not=enzyme-lto-prelink %s

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
