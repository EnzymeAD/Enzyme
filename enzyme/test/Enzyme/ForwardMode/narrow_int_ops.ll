; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,instsimplify,simplifycfg)" -S 2>&1 | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,instsimplify,simplifycfg)" -enzyme-disable-activity-analysis -S | FileCheck %s --check-prefix=NOACT; fi

target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"

declare void @__enzyme_fwddiff(...)
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.abs.i32(i32, i1)
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

define void @step_ops(ptr nocapture %state, ptr nocapture writeonly %out) {
entry:
  %p = getelementptr inbounds { double, i32 }, ptr %state, i64 0, i32 1
  %nfe = load i32, ptr %p, align 4
  %q = sdiv i32 100, %nfe
  %m1 = call i32 @llvm.smax.i32(i32 %q, i32 0)
  %m2 = call i32 @llvm.smin.i32(i32 %m1, i32 50)
  %m3 = call i32 @llvm.umax.i32(i32 %m2, i32 1)
  %m4 = call i32 @llvm.umin.i32(i32 %m3, i32 49)
  %ab = call i32 @llvm.abs.i32(i32 %m4, i1 false)
  %ov = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %ab, i32 1)
  %v = extractvalue { i32, i1 } %ov, 0
  store i32 %v, ptr %p, align 4
  %x = load double, ptr %state, align 8
  %y = fmul double %x, %x
  store double %y, ptr %out, align 8
  ret void
}

define void @copy_i32(ptr nocapture readonly %src, ptr nocapture writeonly %dst) {
entry:
  %v = load i32, ptr %src, align 4
  store i32 %v, ptr %dst, align 4
  ret void
}

define double @extract_struct(ptr nocapture readonly %s, ptr nocapture writeonly %out) {
entry:
  %pair = load { i32, i32 }, ptr %s, align 4
  %a = extractvalue { i32, i32 } %pair, 0
  %b = extractvalue { i32, i32 } %pair, 1
  store i32 %a, ptr %out, align 4
  %bf = sitofp i32 %b to double
  ret double %bf
}

define void @fwd_step(ptr %s, ptr %ds, ptr %o, ptr %do) {
  call void (...) @__enzyme_fwddiff(ptr @step_ops, ptr %s, ptr %ds, ptr %o, ptr %do)
  ret void
}

define void @fwd_step_vec(ptr %s, ptr %ds1, ptr %ds2, ptr %o, ptr %do1, ptr %do2) {
  call void (...) @__enzyme_fwddiff(ptr @step_ops, metadata !"enzyme_width", i64 2, ptr %s, ptr %ds1, ptr %ds2, ptr %o, ptr %do1, ptr %do2)
  ret void
}

define void @fwd_copy(ptr %s, ptr %ds, ptr %d, ptr %dd) {
  call void (...) @__enzyme_fwddiff(ptr @copy_i32, ptr %s, ptr %ds, ptr %d, ptr %dd)
  ret void
}

define double @fwd_extract(ptr %s, ptr %ds, ptr %o, ptr %do) {
  %r = call double (...) @__enzyme_fwddiff(ptr @extract_struct, ptr %s, ptr %ds, ptr %o, ptr %do)
  ret double %r
}

; CHECK-NOT: Mismatched activity

; CHECK-LABEL: define internal void @fwddiffestep_ops(
; CHECK:   %nfe = load i32, ptr %p, align 4
; CHECK-NEXT:   %q = sdiv i32 100, %nfe
; CHECK-NOT: sdiv i32
; CHECK:   %[[DX:.+]] = load double, ptr %"state'", align 8
; CHECK:   %[[X:.+]] = load double, ptr %state, align 8
; CHECK:   %[[MUL:.+]] = fmul fast double %[[DX]], %[[X]]
; CHECK:   %[[ADD:.+]] = fadd fast double %[[MUL]], %[[MUL]]
; CHECK:   store double %[[ADD]], ptr %"out'", align 8
; CHECK:   ret void

; CHECK-LABEL: define internal void @fwddiffe2step_ops(
; CHECK:   %nfe = load i32, ptr %p, align 4
; CHECK-NEXT:   %q = sdiv i32 100, %nfe
; CHECK-NOT: sdiv i32
; CHECK:   ret void

; Untyped i32 copy (e.g. a 4-byte float copy) must still propagate shadow -> shadow.
; CHECK-LABEL: define internal void @fwddiffecopy_i32(
; CHECK:   %"v'ipl" = load i32, ptr %"src'", align 4
; CHECK:   %v = load i32, ptr %src, align 4
; CHECK:   store i32 %"v'ipl", ptr %"dst'", align 4
; CHECK:   store i32 %v, ptr %dst, align 4
; CHECK:   ret void

; Extracting an untyped i32 from { i32, i32 } whose other field is Integer must not
; collide in TypeTree::CanonicalizeInPlace.
; CHECK-LABEL: define internal double @fwddiffeextract_struct(
; CHECK:   %"pair'ipl" = load { i32, i32 }, ptr %"s'", align 4
; CHECK:   %"a'ipev" = extractvalue { i32, i32 } %"pair'ipl", 0
; CHECK:   store i32 %"a'ipev", ptr %"out'", align 4
; CHECK:   ret double 0.000000e+00

; Even with activity analysis disabled, sdiv must never be mirrored onto shadow values.
; NOACT-LABEL: define internal void @fwddiffestep_ops(
; NOACT:   %q = sdiv i32 100, %nfe
; NOACT-NOT: sdiv i32
; NOACT:   ret void
