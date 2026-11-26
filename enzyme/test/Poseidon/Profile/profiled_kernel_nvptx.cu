// REQUIRES: enzyme, nvptx-device-profiler
// DEFINE: %{clang} = %clang -x cuda -nocudainc -nocudalib -S -emit-llvm \
// DEFINE:   -I%poseidon_profile_inc %loadClangPoseidonEnzyme %s
// DEFINE: %{gen} = %{clang} --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -mllvm -poseidon-profile-generate
// DEFINE: %{host} = %{clang} --cuda-host-only -O0 -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin -mllvm -poseidon-cache=%t/c
// RUN: rm -rf %t && mkdir -p %t && echo fatbin > %t/k.fatbin
// RUN: %{gen} -mllvm -poseidon-cache=%t/c -mllvm -poseidon-compilation-id=u -o %t/dev.ll
// RUN: cat %t/c/launches_*.json | FileCheck --check-prefix=MANIFEST %s
// RUN: %{host} -mllvm -poseidon-profile-generate -mllvm -poseidon-compilation-id=u -o %t/host.ll
// RUN: FileCheck --check-prefix=HOST %s < %t/host.ll
// RUN: grep -o 'c"SiteId[^"]*"' %t/dev.ll | sort > %t/dev.headers
// RUN: grep -o 'c"SiteId[^"]*"' %t/host.ll | sort > %t/host.headers
// RUN: count 2 < %t/dev.headers
// RUN: diff %t/dev.headers %t/host.headers
// RUN: %{host} -mllvm -poseidon-profile-generate -mllvm -poseidon-compilation-id=other -o %t/other.ll
// RUN: FileCheck --check-prefix=NONE --implicit-check-not=__poseidon_register_static %s < %t/other.ll
// RUN: %{host} -mllvm -poseidon-compilation-id=u -o %t/plain.ll
// RUN: FileCheck --check-prefix=PLAIN --implicit-check-not=__poseidon_register_static %s < %t/plain.ll
// RUN: %{gen} -DBOUNDS -mllvm -poseidon-cache=%t/bounds -o - | FileCheck --check-prefix=BOUNDS %s
// RUN: %{gen} -DLOCAL -mllvm -poseidon-cache=%t/local -o %t/local.ll
// RUN: %{gen} -DLOCAL -fgpu-rdc -mllvm -poseidon-cache=%t/rdc -o %t/rdc.ll
// RUN: cat %t/local.ll %t/rdc.ll | FileCheck --check-prefix=LOCAL %s
// RUN: %{gen} -DVIEW -mllvm -poseidon-cache=%t/view -mllvm -poseidon-print -o %t/view.ll 2>&1 | FileCheck --check-prefix=VIEW %s
// RUN: cat %t/view/launches_*.json | FileCheck --check-prefix=DESC %s

// Profiled kernels on NVPTX: the host half registers each kernel's header,
// handed over in the launch manifest (a host compile with no manifest for its
// unit, or not generating, registers nothing); the profiled kernel is bounded
// to the largest block unless it declares a bound; a file-local kernel's
// profile is named with the GUID of its identity, alike with -fgpu-rdc; a view
// that is null when empty is still the site's output and takes the seed.

#include "poseidon/poseidon.h"

#define __global__ __attribute__((global))

#if defined(BOUNDS)
POSEIDON_OPTIMIZE __global__ void unbounded(double *y, const double *x, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    y[i] = x[i] * x[i] + y[i];
}

POSEIDON_OPTIMIZE __attribute__((global, launch_bounds(256))) void
bounded(double *y, const double *x, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    y[i] = x[i] * x[i] + y[i];
}
#elif defined(LOCAL)
POSEIDON_OPTIMIZE __global__ static void step(double *x, int n) {
  x[n] = x[n] * x[n] + 1.0;
}

namespace {
POSEIDON_OPTIMIZE __global__ void scale(double *x, int n) {
  x[n] = x[n] / (x[n] + 2.0);
}
} // namespace

POSEIDON_OPTIMIZE __global__ void shared(double *x, int n) {
  x[n] = x[n] * 0.5 + x[n];
}
#elif defined(VIEW)
#define HD __attribute__((host, device))

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  HD DeviceTensor(double *p, int n0, int n1) {
    sizes[0] = n0;
    sizes[1] = n1;
    capacity = n0 * n1;
    data = capacity > 0 ? p : nullptr;
  }
  HD double &operator()(int a, int b) const { return data[a + sizes[0] * b]; }
};

POSEIDON_OPTIMIZE __global__ void param_view(double *y, const double *x, int nd,
                                             int ne) {
  DeviceTensor<2> Y(y, nd, ne);
  int e = __nvvm_read_ptx_sreg_ctaid_x();
  for (int i = __nvvm_read_ptx_sreg_tid_x(); i < nd;
       i += __nvvm_read_ptx_sreg_ntid_x())
    Y(i, e) += x[i] * x[i];
}

template <typename BODY> __global__ static void CuKernel1D(int N, BODY body) {
  int k = __nvvm_read_ptx_sreg_ctaid_x();
  if (k >= N)
    return;
  body(k);
}

void capture_view(int nd, int ne, const double *x, double *y) {
  auto body = [=] HD(int e) POSEIDON_OPTIMIZE {
    DeviceTensor<2> Y(y, nd, ne);
    for (int i = __nvvm_read_ptx_sreg_tid_x(); i < nd;
         i += __nvvm_read_ptx_sreg_ntid_x())
      Y(i, e) += x[i] * x[i];
  };
  void *k = (void *)CuKernel1D<decltype(body)>;
  (void)k;
}
#else
struct dim3 {
  unsigned x, y, z;
};
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);

POSEIDON_OPTIMIZE __global__ void scale(double *x, double a) {
  x[0] = a * x[0] + 1.0;
}

POSEIDON_OPTIMIZE __global__ void norm2(const double *a, double *y) {
  double s = 0.0;
  for (int k = 0; k < 16; ++k)
    s += a[k] * a[k];
  y[0] = s;
}
#endif

// MANIFEST: "profile headers": {
// MANIFEST-NEXT: "_Z5norm2PKdPd": {
// MANIFEST-NEXT: "header": "SiteId = {{[0-9]+}}\nCanonicalHash = {{[0-9a-f]+}}\n",
// MANIFEST-NEXT: "profile": "_Z5norm2PKdPd"
// MANIFEST-NEXT: },
// MANIFEST-NEXT: "_Z5scalePdd": {
// MANIFEST-NEXT: "header": "SiteId = {{[0-9]+}}\nCanonicalHash = {{[0-9a-f]+}}\n",
// MANIFEST-NEXT: "profile": "_Z5scalePdd"
// HOST: @llvm.global_ctors = {{.*}}{ i32 0, ptr @poseidon.register_static._Z5norm2PKdPd, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon.register_static._Z5scalePdd, ptr null }]
// HOST-LABEL: define internal void @poseidon.register_static._Z5norm2PKdPd()
// HOST-NEXT: call void @__poseidon_register_static(ptr @poseidon.site._Z5norm2PKdPd, ptr @poseidon.static._Z5norm2PKdPd)
// HOST-LABEL: define internal void @poseidon.register_static._Z5scalePdd()
// HOST-NEXT: call void @__poseidon_register_static(ptr @poseidon.site._Z5scalePdd, ptr @poseidon.static._Z5scalePdd)
// NONE: @__poseidon_profile_table
// PLAIN: define
// BOUNDS-DAG: define {{.*}}ptx_kernel void @_Z9unboundedPdPKdi({{.*}}) {{.*}}#[[UNB:[0-9]+]]
// BOUNDS-DAG: define {{.*}}ptx_kernel void @_Z7boundedPdPKdi({{.*}}) {{.*}}#[[BND:[0-9]+]]
// BOUNDS-DAG: attributes #[[UNB]] = {{{.*}}"nvvm.maxntid"="1024"
// BOUNDS-DAG: attributes #[[BND]] = {{{.*}}"nvvm.maxntid"="256"
// LOCAL-DAG: c"_ZL4stepPdi~[[STEP:[0-9a-f]+]]\00"
// LOCAL-DAG: c"_ZN12_GLOBAL__N_15scaleEPdi~[[SCALE:[0-9a-f]+]]\00"
// LOCAL-DAG: c"_Z6sharedPdi\00"
// LOCAL: target triple
// LOCAL-DAG: c"_ZL4stepPdi~[[STEP]]\00"
// LOCAL-DAG: c"_ZN12_GLOBAL__N_15scaleEPdi~[[SCALE]]\00"
// LOCAL-DAG: c"_Z6sharedPdi\00"
// VIEW-DAG: [poseidon] _Z10param_viewPdPKdii: site {{[0-9]+}}, 2 shadow buffer(s), output(s) 0{{$}}
// VIEW-DAG: [poseidon] _ZL10CuKernel1DIZ12capture_viewiiPKdPdEUliE_EviT_: site {{[0-9]+}}, 2 shadow buffer(s), output(s) 1[+0]{{$}}
// The pointers among the launch arguments and the seed width of each.
// DESC-DAG: "_Z10param_viewPdPKdii {{[0-9]+}} 4 2 0 1 8 0"
// DESC-DAG: "_ZL10CuKernel1DIZ12capture_viewiiPKdPdEUliE_EviT_ {{[0-9]+}} 2 2 1 1 8 0 off 0 16"
