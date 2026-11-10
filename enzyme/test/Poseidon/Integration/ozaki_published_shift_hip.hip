// ozaki_published_shift.cu compiled as HIP, against the HIP Ozaki-II runtime
// on hipBLAS: the same published shifts and truncation, so the same three rows.
//
// REQUIRES: hip-runtime
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -O2 \
// RUN:     -I%poseidon_runtime_inc %S/ozaki_published_shift.cu -x none \
// RUN:     %poseidon_rt_hip_lib -L%rocm_path/lib -lhipblas -lamdhip64 \
// RUN:     -lstdc++ -lm -o %t.exe
// RUN: %t.exe | FileCheck %S/ozaki_published_shift.cu
