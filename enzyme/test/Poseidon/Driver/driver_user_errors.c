// RUN: not %poseidon_clangxx -poseidon-profile-generate -poseidon-profile-use -c %s -o %t.o 2>&1 | FileCheck --check-prefix=MODES %s
// RUN: not env -u HIP_PATH ROCM_PATH=%t.none %poseidon_clangxx -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc -nogpulib -### %s -o %t.exe 2>&1 | FileCheck --check-prefix=ROCM -DDIR=%t.none %s
// RUN: not env HIP_PATH=%t.none %poseidon_clangxx -poseidon-profile-use -x hip --offload-arch=gfx942 -nogpuinc -nogpulib -### %s -o %t.exe 2>&1 | FileCheck --check-prefix=HIP -DDIR=%t.none %s

// A mistake in what poseidon-clang is given is reported in the form the pass
// reports one in, exit status 1: both halves of the workflow at once, and a
// ROCm environment variable that names no directory, which would otherwise
// leave the link to look for the HIP runtime there.

// MODES: poseidon-clang++: error: Poseidon: -poseidon-profile-generate and -poseidon-profile-use are the two halves of the workflow; pass one
// ROCM: poseidon-clang++: error: Poseidon: ROCM_PATH=[[DIR]] names no directory; set it to the ROCm installation or unset it
// HIP: poseidon-clang++: error: Poseidon: HIP_PATH=[[DIR]] names no directory; set it to the ROCm installation or unset it

int main(void) { return 0; }
