; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -enzyme-max-type-offset=32 -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-preopt=false -enzyme-max-type-offset=32 -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -enzyme-max-type-offset=32 -S | %lli -

declare double @__enzyme_autodiff(ptr, ...)
declare void @abort()

define internal void @push_cursor(ptr %slot) {
entry:
  %cur = load ptr, ptr %slot, align 8
  %next = getelementptr inbounds ptr, ptr %cur, i64 1
  store ptr %next, ptr %slot, align 8
  %chunk = load ptr, ptr %cur, align 8
  store float 2.500000e+00, ptr %chunk, align 4
  ret void
}

define double @compute(double %x, ptr %slot) {
entry:
  call void @push_cursor(ptr %slot)
  %sq = fmul double %x, %x
  ret double %sq
}

define i32 @main() {
entry:
  %chunks = alloca [2 x float], align 4
  %map = alloca [3 x ptr], align 8
  %slot = alloca ptr, align 8
  %chunk0 = getelementptr inbounds [2 x float], ptr %chunks, i64 0, i64 0
  %chunk1 = getelementptr inbounds [2 x float], ptr %chunks, i64 0, i64 1
  %map0 = getelementptr inbounds [3 x ptr], ptr %map, i64 0, i64 0
  %map1 = getelementptr inbounds [3 x ptr], ptr %map, i64 0, i64 1
  store ptr %chunk0, ptr %map0, align 8
  store ptr %chunk1, ptr %map1, align 8
  store ptr %map0, ptr %slot, align 8
  %grad = call double (ptr, ...) @__enzyme_autodiff(ptr @compute, double 3.500000e+00, metadata !"enzyme_const", ptr %slot)
  %diff = fsub double %grad, 7.000000e+00
  %too_low = fcmp olt double %diff, -1.000000e-06
  %too_high = fcmp ogt double %diff, 1.000000e-06
  %bad = or i1 %too_low, %too_high
  br i1 %bad, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffecompute(
; CHECK: %[[M0:.+]] = fmul fast double %differeturn, %x
; CHECK: %[[M1:.+]] = fmul fast double %differeturn, %x
; CHECK: %[[SUM:.+]] = fadd fast double %[[M0]], %[[M1]]
; CHECK: %[[RES:.+]] = insertvalue { double } undef, double %[[SUM]], 0
; CHECK: ret { double } %[[RES]]
