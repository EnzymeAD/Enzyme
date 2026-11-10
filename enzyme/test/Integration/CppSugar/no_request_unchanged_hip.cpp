// HIP device code that asks nothing of Enzyme compiles as it does without the
// plugin, which keeps every function of clang's HIP math header alive in case
// a derivative calls it. Without a pass pipeline the globals that register
// those functions stay in the module; next to an annotation with arguments the
// module still reads back. The header is stood in for, so no ROCm installation
// is needed.

// REQUIRES: amdgpu-registered-target
// RUN: rm -rf %t && split-file %s %t
// RUN: %clang -x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx90a -cuid=0 -O2 -S -include %t/__clang_hip_math.h %t/device.hip -o %t/device.s
// RUN: %clang -x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx90a -cuid=0 -O2 -S -include %t/__clang_hip_math.h %t/device.hip -o %t/device.enzyme.s %loadClangEnzyme
// RUN: diff %t/device.s %t/device.enzyme.s
// RUN: %clang -x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx90a -cuid=0 -O1 -Xclang -disable-llvm-passes -c -emit-llvm -include %t/__clang_hip_math.h %t/device.hip -o %t/device.enzyme.bc %loadClangEnzyme
// RUN: llvm-dis %t/device.enzyme.bc -o - | FileCheck %s
//
// With debug info, no entry for a registration is left in the compile unit.
// RUN: %clang -x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx90a -cuid=0 -O2 -g -S -emit-llvm -include %t/__clang_hip_math.h %t/device.hip -o - %loadClangEnzyme | FileCheck --check-prefix=DEBUG --implicit-check-not=__enzyme_device_math_autoreg %s

// HIP device code is compiled to LTO pre-link bitcode by default, as here with
// -foffload-lto. There Enzyme leaves differentiation to the link unless
// -enzyme-lto-prelink is given, so a derivative another unit asks for may still
// need this unit's math: from LLVM 20, which defers, the unit keeps its math
// functions tagged and its wrappers alive, as before this change. With
// -enzyme-lto-prelink the unit compiles as without the plugin.
// RUN: %clang -x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx90a -cuid=0 -O2 -foffload-lto -c -emit-llvm -include %t/__clang_hip_math.h %t/device.hip -o %t/lto.bc
// RUN: %clang -x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx90a -cuid=0 -O2 -foffload-lto -c -emit-llvm -include %t/__clang_hip_math.h %t/device.hip -o %t/lto.prelink.bc %loadClangEnzyme -mllvm -enzyme-lto-prelink=1
// RUN: llc %t/lto.bc -o %t/lto.s
// RUN: llc %t/lto.prelink.bc -o %t/lto.prelink.s
// RUN: diff %t/lto.s %t/lto.prelink.s
// RUN: if [ %llvmver -ge 20 ]; then %clang -x hip --cuda-device-only -nogpuinc -nogpulib --offload-arch=gfx90a -cuid=0 -O2 -foffload-lto -c -emit-llvm -include %t/__clang_hip_math.h %t/device.hip -o %t/lto.deferred.bc %loadClangEnzyme && llvm-dis %t/lto.deferred.bc -o - | FileCheck %s --check-prefix=DEFERRED; fi

// CHECK: @{{.*}}__enzyme_device_math_autoreg__ZL4sqrtd = internal addrspace(1) global ptr @_ZL4sqrtd
// CHECK: @{{.*}}__enzyme_device_math_autoreg__ZL3cosd = internal addrspace(1) global ptr @_ZL3cosd
// CHECK: @llvm.global.annotations = {{.*}} global [1 x

// DEBUG: !DICompileUnit(

// DEFERRED: define internal {{.*}}double @_ZL3cosd(
// DEFERRED: "enzyme_math"="sqrt"

//--- __clang_hip_math.h
extern "C" __attribute__((device)) double __ocml_sqrt_f64(double);
extern "C" __attribute__((device)) double __ocml_cos_f64(double);
static __attribute__((device)) double sqrt(double x) {
  return __ocml_sqrt_f64(x);
}
static __attribute__((device)) double cos(double x) {
  return __ocml_cos_f64(x);
}

//--- device.hip
__attribute__((global, annotate("tuned", 1))) void
norm1p(double *o, const double *a, int n) {
  int i = __builtin_amdgcn_workitem_id_x();
  if (i < n)
    o[i] = sqrt(a[i] * a[i] + 1.0);
}
