// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print %s -o %t/use.ll 2>&1 | FileCheck --check-prefix=USE %s
// RUN: not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/wrapper -DNOT_A_WRAPPER %s -o /dev/null 2>&1 | FileCheck --check-prefix=WRAPPER %s
// RUN: not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -DNOT_A_WRAPPER %s -o /dev/null 2>&1 | FileCheck --check-prefix=WRAPPER %s

// POSEIDON_OPTIMIZE on the lambda a forall body is written as makes every
// kernel that calls it a site, with the lambda's target. The lambda reaches the
// kernel as one by-value aggregate of DeviceTensors (each laid out as capacity,
// data, sizes), so each data pointer is a field of the
// kernel's parameter: profile generation shadows each of them and seeds the
// one written.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))
#define FOREACH_Y(i, N)                                                        \
  for (int i = __nvvm_read_ptx_sreg_tid_y(); i < N;                            \
       i += __nvvm_read_ptx_sreg_ntid_y())

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  HD double &operator()(int a, int b) const { return data[a + sizes[0] * b]; }
};

template <typename BODY> __attribute__((global)) static void CuKernel2D(int N, BODY body) {
  int k = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_z() +
          __nvvm_read_ptx_sreg_tid_z();
  if (k >= N)
    return;
  body(k);
}

// A kernel that computes more than the lambda cannot be its site.
// WRAPPER: annotated_lambda_nvptx.cu:[[@LINE+3]]:{{[0-9]+}}: error: Poseidon: void Scaled<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, double*, reduce(int, int, int, double*, double*, double*)::'lambda'(int)) calls reduce(int, int, int, double*, double*, double*)::'lambda'(int)::operator()(int) const, which carry POSEIDON_OPTIMIZE, and does floating-point work outside them (in void Scaled<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, double*, reduce(int, int, int, double*, double*, double*)::'lambda'(int))), so it is no wrapper around the call and not their site; move that work out of the kernel, or annotate the kernel instead
// WRAPPER-NOT: error:
// WRAPPER: {{^}}1 error generated when compiling for {{.*}}
template <typename BODY> __attribute__((global)) static void Scaled(int N, double *s, BODY body) {
  int k = __nvvm_read_ptx_sreg_ctaid_x();
  if (k >= N)
    return;
  s[k] = s[k] * 2.0;
  body(k);
}

void reduce(int nq, int ne, int nd, double *Q, double *G, double *y) {
  DeviceTensor<2> QR{nq * ne, Q, {nq, ne}};
  DeviceTensor<2> GR{nq * nd, G, {nq, nd}};
  DeviceTensor<2> Y{nd * ne, y, {nd, ne}};
  auto body = [=] HD POSEIDON_OPTIMIZE_TAU(1e-9) (int e) {
    FOREACH_Y(i, nd) {
      double sum = 0.0;
      for (int p = 0; p < nq; ++p)
        sum += QR(p, e) * GR(p, i);
      Y(i, e) += sum;
    }
  };
  void *k1 = (void *)CuKernel2D<decltype(body)>;
  (void)k1;
#ifdef NOT_A_WRAPPER
  void *k2 = (void *)Scaled<decltype(body)>;
  (void)k2;
#endif
}

// An annotated device function no kernel calls is not a site.
POSEIDON_OPTIMIZE __attribute__((device)) double unused(double x) { return x * x; }

// GEN: annotated_lambda_nvptx.cu:[[@LINE-2]]:{{[0-9]+}}: warning: Poseidon: unused(double) is a device function no kernel of this translation unit calls, so it is left as written; on the device, put POSEIDON_OPTIMIZE on a kernel or on a function a kernel calls
// GEN: [poseidon] block-geometry probe placed in 1 profiled function(s)
// GEN-NEXT: [poseidon] _ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_: site 0, 3 shadow buffer(s), output(s) 1[+64]
// GEN-NOT: [poseidon] {{.*}}Scaled
// GEN: 1 warning generated when compiling for {{.*}}

// The kernel, its two launch arguments, its three pointers (all in argument
// 1), the eight-byte seed of the one written, and where each sits in the
// lambda.
// DESC: "host dispatch": [],
// DESC-NEXT: "profile generation": [
// DESC-NEXT: "_ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_ 0 2 3 1 1 1 0 0 8 off 16 40 64"
// DESC-NEXT: ],
// DESC-NEXT: "profile headers": {
// DESC-NEXT: "_ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_": {
// DESC-NEXT: "header": "SiteId = 0\nCanonicalHash = {{[0-9a-f]+}}\n",
// DESC-NEXT: "profile": "_ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_~620114d1d498cb14"

// The kernel gains the three shadows, in the order of the pointers they
// shadow, and passes the body a copy of the lambda with each data pointer
// replaced by its shadow: the adjoints of Q and G accumulate into the first
// two, and the one of y is read from the third, cleared, and logged as the
// seed of both operations.
// GENIR: @poseidon_site_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body = private unnamed_addr addrspace(1) constant [64 x i8] c"_ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_~620114d1d498cb14\00"
// GENIR: define ptx_kernel void @_ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(i32 [[N:%[0-9]+]], ptr {{.*}}byval(%class.anon){{.*}} [[L:%[0-9]+]], ptr [[DQ:%[0-9]+]], ptr [[DG:%[0-9]+]], ptr [[DY:%[0-9]+]])
// GENIR-NEXT: [[SL:%[0-9]+]] = alloca %class.anon
// GENIR-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[SL]], ptr {{.*}}[[L]], i64 80, i1 false)
// GENIR-NEXT: [[FQ:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SL]], i64 16
// GENIR-NEXT: store ptr [[DQ]], ptr [[FQ]]
// GENIR-NEXT: [[FG:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SL]], i64 40
// GENIR-NEXT: store ptr [[DG]], ptr [[FG]]
// GENIR-NEXT: [[FY:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SL]], i64 64
// GENIR-NEXT: store ptr [[DY]], ptr [[FY]]
// GENIR-NEXT: call fastcc void @diffepreprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body(i32 [[N]], ptr {{.*}}[[L]], ptr [[SL]])
// GENIR-NEXT: ret void
// GENIR: define internal fastcc void @diffepreprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body(i32 {{.*}}%{{[0-9]+}}, ptr {{.*}}%{{[0-9]+}}, ptr {{.*}}[[S:%[0-9]+]])
// GENIR-DAG: call void @poseidonProfileStaticCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), ptr addrspacecast (ptr addrspace(1) @fpprofile_static_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr))
// GENIR-DAG: [[SQP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[S]], i64 16
// GENIR-DAG: [[SQ:%[0-9]+]] = load ptr, ptr [[SQP]]
// GENIR-DAG: [[SGP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[S]], i64 40
// GENIR-DAG: [[SG:%[0-9]+]] = load ptr, ptr [[SGP]]
// GENIR-DAG: [[SYP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[S]], i64 64
// GENIR-DAG: [[SY:%[0-9]+]] = load ptr, ptr [[SYP]]
// GENIR-DAG: [[DQI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SQ]], i64 {{%[0-9]+}}
// GENIR-DAG: atomicrmw fadd ptr [[DQI]], double {{%[0-9]+}} monotonic
// GENIR-DAG: [[DGI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SG]], i64 {{%[0-9]+}}
// GENIR-DAG: atomicrmw fadd ptr [[DGI]], double {{%[0-9]+}} monotonic
// GENIR-DAG: [[DYI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SY]], i64 {{%[0-9]+}}
// GENIR-DAG: [[ADJ:%[0-9]+]] = load double, ptr [[DYI]]
// GENIR-DAG: store double 0.000000e+00, ptr [[DYI]]
// GENIR-DAG: [[G0:%[0-9]+]] = call fastcc double @fixgradient___poseidon_probe_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body_0(double {{%[0-9]+}}, double [[ADJ]])
// GENIR-DAG: atomicrmw fadd ptr [[DYI]], double [[G0]] monotonic
// GENIR-DAG: [[G1IN:%[0-9]+]] = phi double [ [[G0]], %{{[0-9]+}} ]
// GENIR-DAG: call fastcc double @fixgradient___poseidon_probe_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body_1(double {{%[0-9]+}}, double [[G1IN]])
// GENIR: {{^}}}
// GENIR: define internal fastcc double @fixgradient___poseidon_probe_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body_0(double [[V:%[0-9]+]], double [[A:%[0-9]+]])
// GENIR-NEXT: call void @poseidonLogGradCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), i64 0, double [[V]], double [[A]])
// GENIR-NEXT: ret double [[A]]
// GENIR: define internal fastcc double @fixgradient___poseidon_probe_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body_1(double [[V:%[0-9]+]], double [[A:%[0-9]+]])
// GENIR-NEXT: call void @poseidonLogGradCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), i64 1, double [[V]], double [[A]])
// GENIR-NEXT: ret double [[A]]

// USE: [poseidon] site void CuKernel2D<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, reduce(int, int, int, double*, double*, double*)::'lambda'(int)): accuracy target 1e-09
// USE-NEXT: annotated_lambda_nvptx.cu:[[@LINE-106]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_~620114d1d498cb14.fpprofile:
// USE: [poseidon] every site of {{.*}}annotated_lambda_nvptx.cu was left alone; emitting the module as written
// USE-NEXT: 2 warnings generated when compiling for {{.*}}
