// The driver names the translation unit of each invocation: one
// -poseidon-unit reaches both compile jobs of a CUDA and of a HIP compile, the
// way every other Poseidon flag does, and no other job. The name follows from
// the working directory and the arguments, so the same invocation gets it
// again and another one does not. A unit the user names, also for one half
// only, is the only one. An invocation that only links passes the name to no
// job and is not warned about it.
//
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-cache=%t.cache -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o > %t.cuda.jobs 2>&1
// RUN: FileCheck --check-prefix=CUDA %s < %t.cuda.jobs
// RUN: grep -o 'poseidon-unit=[^" ]*' %t.cuda.jobs > %t.cuda.units
// RUN: count 3 < %t.cuda.units
// RUN: sort -u %t.cuda.units | count 1
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-cache=%t.cache -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o 2>&1 \
// RUN:   | grep -o 'poseidon-unit=[^" ]*' | diff %t.cuda.units -
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-cache=%t.cache -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.other.o 2>&1 \
// RUN:   | grep -o 'poseidon-unit=[^" ]*' > %t.other.units
// RUN: count 3 < %t.other.units
// RUN: not diff %t.cuda.units %t.other.units
//
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -x hip --offload-arch=gfx942 --rocm-path=%t.rocm -nogpuinc \
// RUN:     -nogpulib -### %s -o %t.o > %t.hip.jobs 2>&1
// RUN: FileCheck --check-prefix=HIP %s < %t.hip.jobs
// RUN: grep -o 'poseidon-unit=[^" ]*' %t.hip.jobs > %t.hip.units
// RUN: count 3 < %t.hip.units
// RUN: sort -u %t.hip.units | count 1
//
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -poseidon-unit=mine -x cuda --cuda-gpu-arch=%gpu_arch \
// RUN:     --cuda-path=%t.cuda -nocudainc -nocudalib -### %s -o %t.o > %t.mine.jobs 2>&1
// RUN: FileCheck --check-prefix=MINE %s < %t.mine.jobs
// RUN: grep -o 'poseidon-unit=[^" ]*' %t.mine.jobs | count 3
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -poseidon-profile-use \
// RUN:     -Xarch_host -mllvm=-poseidon-unit=mine -x hip --offload-arch=gfx942 \
// RUN:     --rocm-path=%t.rocm -nogpuinc -nogpulib -### %s -o %t.o > %t.host.jobs 2>&1
// RUN: FileCheck --check-prefix=HOST %s < %t.host.jobs
// RUN: grep -o 'poseidon-unit=[^" ]*' %t.host.jobs | count 2
//
// RUN: echo > %t.link.o
// RUN: env POSEIDON_ECHO=1 %poseidon_clangxx -Werror %t.link.o -### -o %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=LINK %s
//
// REQUIRES: poseidon

int main(void) { return 0; }

// CUDA: clang{{.*}} --start-no-unused-arguments -mllvm -poseidon-unit=[[UNIT:[0-9A-F]+]] --end-no-unused-arguments -mllvm -poseidon-profile-use
// CUDA: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}} "-mllvm" "-poseidon-unit=[[UNIT]]"
// CUDA: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-unit=[[UNIT]]"
// CUDA-NOT: poseidon-unit

// HIP: clang{{.*}} --start-no-unused-arguments -Xclang -mllvm -Xclang -poseidon-unit=[[UNIT:[0-9A-F]+]] --end-no-unused-arguments -Xclang -mllvm -Xclang -poseidon-profile-use
// HIP: "-cc1" {{.*}}"-fcuda-is-device"{{.*}} "-mllvm" "-poseidon-unit=[[UNIT]]"
// HIP: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-unit=[[UNIT]]"
// HIP-NOT: poseidon-unit

// MINE: clang{{.*}} -mllvm -poseidon-unit=mine
// MINE: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}} "-mllvm" "-poseidon-unit=mine"
// MINE: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-unit=mine"
// MINE-NOT: poseidon-unit

// HOST: clang{{.*}} -Xarch_host -mllvm=-poseidon-unit=mine
// HOST-NOT: poseidon-unit
// HOST: "-cc1" "-triple" "x86_64-{{[^"]+}}"{{.*}} "-mllvm" "-poseidon-unit=mine"
// HOST-NOT: poseidon-unit

// LINK: clang{{.*}} --start-no-unused-arguments -mllvm -poseidon-unit={{[0-9A-F]+}} --end-no-unused-arguments
// LINK-NOT: poseidon-unit
// LINK-NOT: unused
