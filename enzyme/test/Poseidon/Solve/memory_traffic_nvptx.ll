; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-print -S
; DEFINE: %{prof} = -poseidon-profile-use=%S/../Inputs/memory_traffic
; DEFINE: %{stream} = -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_stream.csv
; RUN: rm -rf %t && mkdir %t %t/untraced
; RUN: %{opt} %{prof} %{stream} -poseidon-tau=1e-6 -poseidon-cache=%t/c1 -poseidon-explain=%t/tau \
; RUN:   -o %t/tau.ll 2>&1 | tee %t/tau.log | FileCheck --check-prefix=STREAM %s
; RUN: FileCheck --check-prefix=STREAM-IR %s < %t/tau.ll
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/tau.log %t/tau | FileCheck --check-prefix=TAU-JSON %s
; RUN: FileCheck --check-prefix=SHORT-TIME %s < %t/tau/_Z9shortpolyiPKdPd.explain.json
; RUN: FileCheck --check-prefix=MID-TIME %s < %t/tau/_Z7midpolyiPKdPd.explain.json
; RUN: %{opt} %{prof} %{stream} -poseidon-tau=1e-6 -poseidon-early-prune -poseidon-cache=%t/c2 \
; RUN:   -o /dev/null 2>&1 | FileCheck --check-prefix=STREAM %s
; RUN: %{opt} %{prof} %{stream} -poseidon-cost-budget=-0.5 -poseidon-cache=%t/c3 -poseidon-explain=%t/budget \
; RUN:   -o /dev/null 2>&1 | tee %t/budget.log | FileCheck --check-prefix=BUDGET %s
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/budget.log %t/budget | FileCheck --check-prefix=BUDGET-JSON %s
; RUN: %{opt} %{prof} -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; RUN:   -poseidon-cache=%t/c4 -poseidon-explain=%t/arith -o %t/arith.ll 2>&1 | tee %t/arith.log | FileCheck --check-prefix=ARITH %s
; RUN: FileCheck --check-prefix=ARITH-IR %s < %t/arith.ll
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/arith.log %t/arith | FileCheck --check-prefix=ARITH-JSON %s
; RUN: FileCheck --check-prefix=NO-TIME %s < %t/arith/_Z7midpolyiPKdPd.explain.json
; RUN: for f in %S/../Inputs/memory_traffic/*.fpprofile; do \
; RUN:   grep -v '^Traffic' $f > %t/untraced/$(basename $f); done
; RUN: %{opt} -poseidon-profile-use=%t/untraced %{stream} -poseidon-tau=1e-6 -poseidon-cache=%t/c5 \
; RUN:   -o /dev/null 2>&1 | FileCheck --check-prefix=UNTRACED %s
; RUN: not %{opt} %{prof} -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_stream_partial.csv \
; RUN:   -poseidon-cache=%t/c6 -o /dev/null 2>&1 | FileCheck --check-prefix=PARTIAL %s

; Pricing a kernel site by its arithmetic A and its memory traffic M combined,
; (A^p + M^p)^(1/p) with p = 64; the IR's three kernels stream 4096 doubles in
; and out with 2, 4 and 32 FP64 multiply-adds per element (M = 8.39e6 against
; A = 4.19e6, 8.39e6, 6.71e7). A rewrite selected for its cost is taken only
; when the combined time falls: the memory-bound kernel stays as written, the
; balanced one's gain lies inside the band, the arithmetic-bound one runs at
; the memory time; the exact and the pruned solver agree. Under a budget that
; the code as written exceeds, the selected point is taken. Without stream rows
; or traffic records the arithmetic decides and the explanation has no time; a
; model with some stream rows stops the compile.

; STREAM: [poseidon] preprocess__Z9shortpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=5.631250e+03 A0=4.485301e-14 selected cost=-8.730469e-01 accCost=1.209417e-04 rel=2.147688e-08
; STREAM-NEXT: [poseidon] preprocess__Z9shortpolyiPKdPd_poseidon_body: arithmetic 4.194304e+06 -> 5.324800e+05, memory [[M:8.388608e\+06]], combined (pnorm 6.400000e+01) [[M]] -> [[M]], modelled speedup 1.000000e+00, band 5.000000e-02: not cheaper than the code as written, which stays
; STREAM-NOT: Applying solution
; STREAM: [poseidon] pin: {{$}}
; STREAM-NOT: Applying solution
; STREAM: [poseidon] preprocess__Z7midpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=5.779927e+03 A0=8.459899e-14 selected cost=-9.355469e-01 accCost=1.057706e-04 rel=1.829965e-08
; STREAM-NEXT: [poseidon] preprocess__Z7midpolyiPKdPd_poseidon_body: arithmetic [[M]] -> 5.406720e+05, memory [[M]], combined (pnorm 6.400000e+01) 8.479954e+06 -> [[M]], modelled speedup 1.010889e+00, band 5.000000e-02: taken, its modelled gain inside the band
; STREAM: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; STREAM: [poseidon] pin: _Z7midpolyiPKdPd:P0=All FP64(0%) + FP32(100%)
; STREAM: [poseidon] preprocess__Z8longpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=7.523840e+02 A0=7.077672e-15 selected cost=-9.902344e-01 accCost=1.261380e-05 rel=1.676511e-08
; STREAM-NEXT: [poseidon] preprocess__Z8longpolyiPKdPd_poseidon_body: arithmetic [[A0L:6.710886e\+07]] -> 6.553600e+05, memory [[M]], combined (pnorm 6.400000e+01) [[A0L]] -> [[M]], modelled speedup 8.000000e+00, band 5.000000e-02: taken
; STREAM: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; STREAM: [poseidon] pin: _Z8longpolyiPKdPd:P0=All FP64(0%) + FP32(100%)

; STREAM-IR-LABEL: define {{.*}} @_Z9shortpolyiPKdPd(
; STREAM-IR-SAME: i32 {{.*}}[[SN:%[0-9]+]], ptr {{.*}}[[SX:%[0-9]+]], ptr {{.*}}[[SY:%[0-9]+]])
; STREAM-IR-NEXT: entry:
; STREAM-IR-NEXT: call void @_Z9shortpolyiPKdPd_poseidon_body(i32 [[SN]], ptr [[SX]], ptr [[SY]])
; STREAM-IR-NEXT: ret void
; STREAM-IR-LABEL: define {{.*}} @_Z7midpolyiPKdPd(
; STREAM-IR-SAME: i32 {{.*}}[[MN:%[0-9]+]], ptr {{.*}}[[MX:%[0-9]+]], ptr {{.*}}[[MY:%[0-9]+]])
; STREAM-IR-NEXT: entry:
; STREAM-IR-NEXT: call void @preprocess__Z7midpolyiPKdPd_poseidon_body(i32 [[MN]], ptr [[MX]], ptr [[MY]])
; STREAM-IR-NEXT: ret void
; STREAM-IR-LABEL: define {{.*}} @_Z8longpolyiPKdPd(
; STREAM-IR-SAME: i32 {{.*}}[[LN:%[0-9]+]], ptr {{.*}}[[LX:%[0-9]+]], ptr {{.*}}[[LY:%[0-9]+]])
; STREAM-IR-NEXT: entry:
; STREAM-IR-NEXT: call void @preprocess__Z8longpolyiPKdPd_poseidon_body(i32 [[LN]], ptr [[LX]], ptr [[LY]])
; STREAM-IR-NEXT: ret void
; STREAM-IR-NOT: define internal void @preprocess__Z9shortpolyiPKdPd_poseidon_body(
; STREAM-IR-LABEL: define internal void @preprocess__Z7midpolyiPKdPd_poseidon_body(
; STREAM-IR-SAME: i32 {{.*}}%{{[0-9]+}}, ptr {{.*}}[[PX:%[0-9]+]], ptr {{.*}}[[PY:%[0-9]+]])
; STREAM-IR: [[XP:%[0-9]+]] = getelementptr inbounds [8 x i8], ptr [[PX]], i64 [[I:%[0-9]+]]
; STREAM-IR-NEXT: [[X:%[0-9]+]] = load double, ptr [[XP]],
; STREAM-IR-NEXT: [[XF:%.+]] = fptrunc double [[X]] to float
; STREAM-IR-NEXT: [[A:%[0-9]+]] = fadd float [[XF]], 5.000000e-01
; STREAM-IR-NEXT: [[B:%[0-9]+]] = tail call float @llvm.fmuladd.f32(float [[A]], float [[XF]], float {{f0x3EAAAAAB|0x3FD5555560000000}})
; STREAM-IR-NEXT: [[C:%[0-9]+]] = tail call float @llvm.fmuladd.f32(float [[B]], float [[XF]], float 2.500000e-01)
; STREAM-IR-NEXT: [[D:%[0-9]+]] = tail call float @llvm.fmuladd.f32(float [[C]], float [[XF]], float {{2.000000e-01|0x3FC99999A0000000}})
; STREAM-IR-NEXT: [[R:%[0-9]+]] = fpext float [[D]] to double
; STREAM-IR-NEXT: [[YP:%[0-9]+]] = getelementptr inbounds [8 x i8], ptr [[PY]], i64 [[I]]
; STREAM-IR-NEXT: store double [[R]], ptr [[YP]],
; STREAM-IR-LABEL: define internal void @preprocess__Z8longpolyiPKdPd_poseidon_body(
; STREAM-IR-SAME: i32 {{.*}}%{{[0-9]+}}, ptr {{.*}}[[QX:%[0-9]+]], ptr {{.*}}[[QY:%[0-9]+]])
; STREAM-IR: [[LXP:%[0-9]+]] = getelementptr inbounds [8 x i8], ptr [[QX]], i64 [[J:%[0-9]+]]
; STREAM-IR-NEXT: [[LXV:%[0-9]+]] = load double, ptr [[LXP]],
; STREAM-IR-NEXT: [[LXF:%.+]] = fptrunc double [[LXV]] to float
; STREAM-IR-NEXT: [[LA:%[0-9]+]] = fadd float [[LXF]], 5.000000e-01
; STREAM-IR-NEXT: {{%[0-9]+}} = tail call float @llvm.fmuladd.f32(float [[LA]], float [[LXF]], float {{f0x3EAAAAAB|0x3FD5555560000000}})
; STREAM-IR-NOT: double
; STREAM-IR: [[LZ:%[0-9]+]] = tail call float @llvm.fmuladd.f32(float {{%[0-9]+}}, float [[LXF]], float {{f0x3CF83E10|0x3F9F07C200000000}})
; STREAM-IR-NEXT: [[LR:%[0-9]+]] = fpext float [[LZ]] to double
; STREAM-IR-NEXT: [[LYP:%[0-9]+]] = getelementptr inbounds [8 x i8], ptr [[QY]], i64 [[J]]
; STREAM-IR-NEXT: store double [[LR]], ptr [[LYP]],

; TAU-JSON: preprocess__Z7midpolyiPKdPd_poseidon_body: rewritten{{$}}
; TAU-JSON-NEXT: preprocess__Z8longpolyiPKdPd_poseidon_body: rewritten{{$}}
; TAU-JSON-NEXT: preprocess__Z9shortpolyiPKdPd_poseidon_body: as_written; the selected frontier point is not faster than the code as written once the memory traffic is priced{{$}}

; SHORT-TIME:      "result": {
; SHORT-TIME-NEXT:   "applied": [],
; SHORT-TIME:        "time": {
; SHORT-TIME-NEXT:     "arithmetic_as_written": 4194304,
; SHORT-TIME-NEXT:     "arithmetic_selected": 532480,
; SHORT-TIME-NEXT:     "band": 0.05{{[0-9]*}},
; SHORT-TIME-NEXT:     "combined_as_written": 8388608,
; SHORT-TIME-NEXT:     "combined_selected": 8388608,
; SHORT-TIME-NEXT:     "inside_band": false,
; SHORT-TIME-NEXT:     "memory": 8388608,
; SHORT-TIME-NEXT:     "modelled_speedup": 1,
; SHORT-TIME-NEXT:     "pnorm": 64,
; SHORT-TIME-NEXT:     "taken": false
; SHORT-TIME-NEXT:   }

; MID-TIME:      "result": {
; MID-TIME-NEXT:   "applied": [
; MID-TIME-NEXT:     "P0_0"
; MID-TIME-NEXT:   ],
; MID-TIME:        "time": {
; MID-TIME-NEXT:     "arithmetic_as_written": 8388608,
; MID-TIME-NEXT:     "arithmetic_selected": 540672,
; MID-TIME-NEXT:     "band": 0.05{{[0-9]*}},
; MID-TIME-NEXT:     "combined_as_written": 8479953.95{{[0-9]+}},
; MID-TIME-NEXT:     "combined_selected": 8388608,
; MID-TIME-NEXT:     "inside_band": true,
; MID-TIME-NEXT:     "memory": 8388608,
; MID-TIME-NEXT:     "modelled_speedup": 1.01088928{{[0-9]+}},
; MID-TIME-NEXT:     "pnorm": 64,
; MID-TIME-NEXT:     "taken": true
; MID-TIME-NEXT:   }

; BUDGET: Computation cost budget used: -8.730469e-01
; BUDGET-NEXT: [poseidon] preprocess__Z9shortpolyiPKdPd_poseidon_body: arithmetic 4.194304e+06 -> 5.324800e+05, memory [[M:8.388608e\+06]], combined (pnorm 6.400000e+01) [[M]] -> [[M]], modelled speedup 1.000000e+00, band 5.000000e-02: taken for the cost budget, which the code as written exceeds
; BUDGET: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; BUDGET: [poseidon] pin: _Z9shortpolyiPKdPd:P0=All FP64(0%) + FP32(100%)
; BUDGET: Computation cost budget used: -6.860352e-01
; BUDGET-NEXT: [poseidon] preprocess__Z7midpolyiPKdPd_poseidon_body: arithmetic [[M]] -> 2.633728e+06, memory [[M]], combined (pnorm 6.400000e+01) 8.479954e+06 -> [[M]], modelled speedup 1.010889e+00, band 5.000000e-02: taken, its modelled gain inside the band
; BUDGET: [poseidon] pin: _Z7midpolyiPKdPd:P0=All FP64(30%) + FP32(70%)
; BUDGET: Computation cost budget used: -5.233154e-01
; BUDGET-NEXT: [poseidon] preprocess__Z8longpolyiPKdPd_poseidon_body: arithmetic 6.710886e+07 -> [[A1L:3.198976e\+07]], memory [[M]], combined (pnorm 6.400000e+01) 6.710886e+07 -> [[A1L]], modelled speedup 2.097823e+00, band 5.000000e-02: taken
; BUDGET: [poseidon] pin: _Z8longpolyiPKdPd:P0=All FP64(40%) + Expansion2(60%)

; BUDGET-JSON: preprocess__Z7midpolyiPKdPd_poseidon_body: rewritten{{$}}
; BUDGET-JSON-NEXT: preprocess__Z8longpolyiPKdPd_poseidon_body: rewritten{{$}}
; BUDGET-JSON-NEXT: preprocess__Z9shortpolyiPKdPd_poseidon_body: rewritten{{$}}

; ARITH: [poseidon] preprocess__Z9shortpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=5.631250e+03 A0=4.485301e-14 selected cost=-8.730469e-01 accCost=1.209417e-04 rel=2.147688e-08
; ARITH-NEXT: [poseidon] preprocess__Z9shortpolyiPKdPd_poseidon_body: memory traffic not priced, the cost model has no stream rows; the arithmetic decides
; ARITH: [poseidon] pin: _Z9shortpolyiPKdPd:P0=All FP64(0%) + FP32(100%)
; ARITH: [poseidon] preprocess__Z7midpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=5.779927e+03 A0=8.459899e-14 selected cost=-9.355469e-01 accCost=1.057706e-04 rel=1.829965e-08
; ARITH-NEXT: [poseidon] preprocess__Z7midpolyiPKdPd_poseidon_body: memory traffic not priced, the cost model has no stream rows; the arithmetic decides
; ARITH: [poseidon] pin: _Z7midpolyiPKdPd:P0=All FP64(0%) + FP32(100%)
; ARITH: [poseidon] preprocess__Z8longpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=7.523840e+02 A0=7.077672e-15 selected cost=-9.902344e-01 accCost=1.261380e-05 rel=1.676511e-08
; ARITH-NEXT: [poseidon] preprocess__Z8longpolyiPKdPd_poseidon_body: memory traffic not priced, the cost model has no stream rows; the arithmetic decides
; ARITH: [poseidon] pin: _Z8longpolyiPKdPd:P0=All FP64(0%) + FP32(100%)

; ARITH-IR-LABEL: define {{.*}} @_Z9shortpolyiPKdPd(
; ARITH-IR-SAME: i32 {{.*}}[[SN:%[0-9]+]], ptr {{.*}}[[SX:%[0-9]+]], ptr {{.*}}[[SY:%[0-9]+]])
; ARITH-IR-NEXT: entry:
; ARITH-IR-NEXT: call void @preprocess__Z9shortpolyiPKdPd_poseidon_body(i32 [[SN]], ptr [[SX]], ptr [[SY]])
; ARITH-IR-NEXT: ret void
; ARITH-IR-LABEL: define internal void @preprocess__Z9shortpolyiPKdPd_poseidon_body(
; ARITH-IR-SAME: i32 {{.*}}%{{[0-9]+}}, ptr {{.*}}[[PX:%[0-9]+]], ptr {{.*}}[[PY:%[0-9]+]])
; ARITH-IR: [[YP:%[0-9]+]] = getelementptr inbounds [8 x i8], ptr [[PY]], i64 [[I:%[0-9]+]]
; ARITH-IR-NEXT: [[XP:%[0-9]+]] = getelementptr inbounds [8 x i8], ptr [[PX]], i64 [[I]]
; ARITH-IR-NEXT: [[X:%[0-9]+]] = load double, ptr [[XP]],
; ARITH-IR-NEXT: [[XF:%.+]] = fptrunc double [[X]] to float
; ARITH-IR-NEXT: [[A:%[0-9]+]] = fadd float [[XF]], 5.000000e-01
; ARITH-IR-NEXT: [[B:%[0-9]+]] = tail call float @llvm.fmuladd.f32(float [[A]], float [[XF]], float {{f0x3EAAAAAB|0x3FD5555560000000}})
; ARITH-IR-NEXT: [[R:%[0-9]+]] = fpext float [[B]] to double
; ARITH-IR-NEXT: store double [[R]], ptr [[YP]],

; ARITH-JSON: preprocess__Z7midpolyiPKdPd_poseidon_body: rewritten{{$}}
; ARITH-JSON-NEXT: preprocess__Z8longpolyiPKdPd_poseidon_body: rewritten{{$}}
; ARITH-JSON-NEXT: preprocess__Z9shortpolyiPKdPd_poseidon_body: rewritten{{$}}

; NO-TIME:     "result": {
; NO-TIME-NOT: "time"
; NO-TIME:     "subgraphs"

; UNTRACED: [poseidon] preprocess__Z9shortpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=5.631250e+03 A0=4.485301e-14 selected cost=-8.730469e-01 accCost=1.209417e-04 rel=2.147688e-08
; UNTRACED-NEXT: [poseidon] preprocess__Z9shortpolyiPKdPd_poseidon_body: memory traffic not priced, its profile records no memory traffic; the arithmetic decides
; UNTRACED: [poseidon] pin: _Z9shortpolyiPKdPd:P0=All FP64(0%) + FP32(100%)
; UNTRACED: [poseidon] preprocess__Z7midpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=5.779927e+03 A0=8.459899e-14 selected cost=-9.355469e-01 accCost=1.057706e-04 rel=1.829965e-08
; UNTRACED-NEXT: [poseidon] preprocess__Z7midpolyiPKdPd_poseidon_body: memory traffic not priced, its profile records no memory traffic; the arithmetic decides
; UNTRACED: [poseidon] pin: _Z7midpolyiPKdPd:P0=All FP64(0%) + FP32(100%)
; UNTRACED: [poseidon] preprocess__Z8longpolyiPKdPd_poseidon_body: tau=1.000000e-06 S=7.523840e+02 A0=7.077672e-15 selected cost=-9.902344e-01 accCost=1.261380e-05 rel=1.676511e-08
; UNTRACED-NEXT: [poseidon] preprocess__Z8longpolyiPKdPd_poseidon_body: memory traffic not priced, its profile records no memory traffic; the arithmetic decides
; UNTRACED: [poseidon] pin: _Z8longpolyiPKdPd:P0=All FP64(0%) + FP32(100%)

; PARTIAL: error: <unknown>:0:0: in function preprocess__Z9shortpolyiPKdPd_poseidon_body void (i32, ptr, ptr): Poseidon: cost model {{.*}}/cm_sm_120_synthetic_stream_partial.csv has no row 'stream_combine,band'; measure it with poseidon-calibrate{{$}}

source_filename = "poly3.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = hidden unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = hidden unnamed_addr constant [9 x i8] c"poly3.cu\00", section "llvm.metadata"
@.args = hidden unnamed_addr constant { i32 } zeroinitializer, section "llvm.metadata"
@llvm.global.annotations = appending global [3 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z9shortpolyiPKdPd, ptr @.str, ptr @.str1, i32 17, ptr @.args }, { ptr, ptr, ptr, i32, ptr } { ptr @_Z7midpolyiPKdPd, ptr @.str, ptr @.str1, i32 23, ptr @.args }, { ptr, ptr, ptr, i32, ptr } { ptr @_Z8longpolyiPKdPd, ptr @.str, ptr @.str1, i32 29, ptr @.args }], section "llvm.metadata"

define dso_local ptx_kernel void @_Z9shortpolyiPKdPd(i32 noundef %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = icmp slt i32 %8, %0
  br i1 %9, label %10, label %17

10:                                               ; preds = %3
  %11 = sext i32 %8 to i64
  %12 = getelementptr inbounds [8 x i8], ptr %2, i64 %11
  %13 = getelementptr inbounds [8 x i8], ptr %1, i64 %11
  %14 = load double, ptr %13, align 8
  %15 = fadd double %14, 5.000000e-01
  %16 = tail call noundef double @llvm.fmuladd.f64(double %15, double %14, double 0x3FD5555555555555)
  store double %16, ptr %12, align 8
  br label %17

17:                                               ; preds = %10, %3
  ret void
}
define dso_local ptx_kernel void @_Z7midpolyiPKdPd(i32 noundef %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = icmp slt i32 %8, %0
  br i1 %9, label %10, label %19

10:                                               ; preds = %3
  %11 = sext i32 %8 to i64
  %12 = getelementptr inbounds [8 x i8], ptr %1, i64 %11
  %13 = load double, ptr %12, align 8
  %14 = fadd double %13, 5.000000e-01
  %15 = tail call double @llvm.fmuladd.f64(double %14, double %13, double 0x3FD5555555555555)
  %16 = tail call double @llvm.fmuladd.f64(double %15, double %13, double 2.500000e-01)
  %17 = tail call noundef double @llvm.fmuladd.f64(double %16, double %13, double 2.000000e-01)
  %18 = getelementptr inbounds [8 x i8], ptr %2, i64 %11
  store double %17, ptr %18, align 8
  br label %19

19:                                               ; preds = %10, %3
  ret void
}
define dso_local ptx_kernel void @_Z8longpolyiPKdPd(i32 noundef %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = icmp slt i32 %8, %0
  br i1 %9, label %10, label %47

10:                                               ; preds = %3
  %11 = sext i32 %8 to i64
  %12 = getelementptr inbounds [8 x i8], ptr %1, i64 %11
  %13 = load double, ptr %12, align 8
  %14 = fadd double %13, 5.000000e-01
  %15 = tail call double @llvm.fmuladd.f64(double %14, double %13, double 0x3FD5555555555555)
  %16 = tail call double @llvm.fmuladd.f64(double %15, double %13, double 2.500000e-01)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double 2.000000e-01)
  %18 = tail call double @llvm.fmuladd.f64(double %17, double %13, double 0x3FC5555555555555)
  %19 = tail call double @llvm.fmuladd.f64(double %18, double %13, double 0x3FC2492492492492)
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %13, double 1.250000e-01)
  %21 = tail call double @llvm.fmuladd.f64(double %20, double %13, double 0x3FBC71C71C71C71C)
  %22 = tail call double @llvm.fmuladd.f64(double %21, double %13, double 1.000000e-01)
  %23 = tail call double @llvm.fmuladd.f64(double %22, double %13, double 0x3FB745D1745D1746)
  %24 = tail call double @llvm.fmuladd.f64(double %23, double %13, double 0x3FB5555555555555)
  %25 = tail call double @llvm.fmuladd.f64(double %24, double %13, double 0x3FB3B13B13B13B14)
  %26 = tail call double @llvm.fmuladd.f64(double %25, double %13, double 0x3FB2492492492492)
  %27 = tail call double @llvm.fmuladd.f64(double %26, double %13, double 0x3FB1111111111111)
  %28 = tail call double @llvm.fmuladd.f64(double %27, double %13, double 6.250000e-02)
  %29 = tail call double @llvm.fmuladd.f64(double %28, double %13, double 0x3FAE1E1E1E1E1E1E)
  %30 = tail call double @llvm.fmuladd.f64(double %29, double %13, double 0x3FAC71C71C71C71C)
  %31 = tail call double @llvm.fmuladd.f64(double %30, double %13, double 0x3FAAF286BCA1AF28)
  %32 = tail call double @llvm.fmuladd.f64(double %31, double %13, double 5.000000e-02)
  %33 = tail call double @llvm.fmuladd.f64(double %32, double %13, double 0x3FA8618618618618)
  %34 = tail call double @llvm.fmuladd.f64(double %33, double %13, double 0x3FA745D1745D1746)
  %35 = tail call double @llvm.fmuladd.f64(double %34, double %13, double 0x3FA642C8590B2164)
  %36 = tail call double @llvm.fmuladd.f64(double %35, double %13, double 0x3FA5555555555555)
  %37 = tail call double @llvm.fmuladd.f64(double %36, double %13, double 4.000000e-02)
  %38 = tail call double @llvm.fmuladd.f64(double %37, double %13, double 0x3FA3B13B13B13B14)
  %39 = tail call double @llvm.fmuladd.f64(double %38, double %13, double 0x3FA2F684BDA12F68)
  %40 = tail call double @llvm.fmuladd.f64(double %39, double %13, double 0x3FA2492492492492)
  %41 = tail call double @llvm.fmuladd.f64(double %40, double %13, double 0x3FA1A7B9611A7B96)
  %42 = tail call double @llvm.fmuladd.f64(double %41, double %13, double 0x3FA1111111111111)
  %43 = tail call double @llvm.fmuladd.f64(double %42, double %13, double 0x3FA0842108421084)
  %44 = tail call double @llvm.fmuladd.f64(double %43, double %13, double 3.125000e-02)
  %45 = tail call noundef double @llvm.fmuladd.f64(double %44, double %13, double 0x3F9F07C1F07C1F08)
  %46 = getelementptr inbounds [8 x i8], ptr %2, i64 %11
  store double %45, ptr %46, align 8
  br label %47

47:                                               ; preds = %10, %3
  ret void
}
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
