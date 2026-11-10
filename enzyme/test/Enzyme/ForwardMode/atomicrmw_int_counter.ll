; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@thread_counter = global i32 0, align 4
@max_seen = global i64 0, align 8

declare void @__enzyme_fwddiff(ptr, ...)

define void @preprocess_rhs(ptr noalias nocapture readonly %in, ptr noalias nocapture writeonly %out, i32 %tid) {
entry:
  %old = atomicrmw add ptr @thread_counter, i32 1 monotonic, align 4
  %cur = load atomic i32, ptr @thread_counter monotonic, align 4
  %tid64 = sext i32 %tid to i64
  %prev_max = atomicrmw max ptr @max_seen, i64 %tid64 monotonic, align 8
  %idx = zext i32 %tid to i64
  %p_in = getelementptr inbounds double, ptr %in, i64 %idx
  %val = load double, ptr %p_in, align 8
  %sq = fmul double %val, %val
  %p_out = getelementptr inbounds double, ptr %out, i64 %idx
  store double %sq, ptr %p_out, align 8
  ret void
}

define void @caller(ptr %in, ptr %din, ptr %out, ptr %dout, i32 %tid) {
entry:
  call void (ptr, ...) @__enzyme_fwddiff(ptr @preprocess_rhs, metadata !"enzyme_dup", ptr %in, ptr %din, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", i32 %tid)
  ret void
}

; CHECK: define internal void @fwddiffepreprocess_rhs(ptr noalias nocapture readonly %in, ptr nocapture %"in'", ptr noalias nocapture writeonly %out, ptr nocapture %"out'", i32 %tid)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %old = atomicrmw add ptr @thread_counter, i32 1 monotonic, align 4
; CHECK-NEXT:   %cur = load atomic i32, ptr @thread_counter monotonic, align 4
; CHECK-NEXT:   %tid64 = sext i32 %tid to i64
; CHECK-NEXT:   %prev_max = atomicrmw max ptr @max_seen, i64 %tid64 monotonic, align 8
; CHECK-NEXT:   %idx = zext i32 %tid to i64
; CHECK-NEXT:   %[[GIN:.+]] = getelementptr inbounds double, ptr %"in'", i64 %idx
; CHECK-NEXT:   %p_in = getelementptr inbounds double, ptr %in, i64 %idx
; CHECK-NEXT:   %[[DIN:.+]] = load double, ptr %[[GIN]], align 8
; CHECK-NEXT:   %val = load double, ptr %p_in, align 8
; CHECK-NEXT:   %sq = fmul double %val, %val
; CHECK-NEXT:   %[[M1:.+]] = fmul fast double %[[DIN]], %val
; CHECK-NEXT:   %[[M2:.+]] = fmul fast double %[[DIN]], %val
; CHECK-NEXT:   %[[DSQ:.+]] = fadd fast double %[[M1]], %[[M2]]
; CHECK-NEXT:   %[[GOUT:.+]] = getelementptr inbounds double, ptr %"out'", i64 %idx
; CHECK-NEXT:   %p_out = getelementptr inbounds double, ptr %out, i64 %idx
; CHECK-NEXT:   store double %[[DSQ]], ptr %[[GOUT]], align 8
; CHECK-NEXT:   store double %sq, ptr %p_out, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
