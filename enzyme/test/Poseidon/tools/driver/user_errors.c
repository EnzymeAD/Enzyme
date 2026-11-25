// RUN: (%poseidon_clangxx -poseidon-profile-generate -poseidon-profile-use -c %s -o %t.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=MODES %s
// RUN: (env -u HIP_PATH ROCM_PATH=%t.none %poseidon_clangxx -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc -nogpulib -### %s -o %t.exe 2>&1; echo "exit status $?") | FileCheck --check-prefix=ROCM -DDIR=%t.none %s
// RUN: (env HIP_PATH=%t.none %poseidon_clangxx -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc -nogpulib -### %s -o %t.exe 2>&1; echo "exit status $?") | FileCheck --check-prefix=HIP -DDIR=%t.none %s

// A mistake in what poseidon-clang is given is reported in the form the pass
// reports one in, exit status 1: both halves of the workflow at once, and a
// ROCm environment variable that names no directory, which would otherwise
// leave the link to look for the HIP runtime there.

// MODES: poseidon-clang++: error: Poseidon: -poseidon-profile-generate and -poseidon-profile-use are the two halves of the workflow; pass one{{$}}
// MODES-NEXT: exit status 1
// ROCM: poseidon-clang++: error: Poseidon: ROCM_PATH=[[DIR]] names no directory; set it to the ROCm installation or unset it{{$}}
// ROCM-NEXT: exit status 1
// HIP: poseidon-clang++: error: Poseidon: HIP_PATH=[[DIR]] names no directory; set it to the ROCm installation or unset it{{$}}
// HIP-NEXT: exit status 1

int main(void) { return 0; }
