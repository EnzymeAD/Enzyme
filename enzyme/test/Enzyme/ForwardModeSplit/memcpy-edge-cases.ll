; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare double @__enzyme_fwdsplit(...)
declare void @__enzyme_fwdsplit.v(...)

@const_dbl = private unnamed_addr constant double 42.0, align 8

; 1. ForwardModeSplit memcpy from constant global must zero the float shadow.
define double @fwdsplit_const(ptr %dst) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 @const_dbl, i64 8, i1 false)
  %v = load double, ptr %dst, align 8
  ret double %v
}

define double @dfwdsplit_const(ptr %dst, ptr %ddst) {
entry:
  %r = call double (...) @__enzyme_fwdsplit(ptr @fwdsplit_const, metadata !"enzyme_dup", ptr %dst, ptr %ddst, ptr null)
  ret double %r
}

; 2. ForwardModeSplit + runtimeActivity dynamic memcpy:
;    Must check both dst_inactive and src_inactive, zeroing shadow when src is inactive and dst is active.
define void @fwdsplit_ra_dyn(ptr %dst, ptr %src) {
entry:
  %d1 = load double, ptr %dst, align 8, !enzyme_type !0
  %d2 = load double, ptr %src, align 8, !enzyme_type !0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 8, i1 false)
  ret void
}

define void @dfwdsplit_ra_dyn(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc) {
entry:
  call void (...) @__enzyme_fwdsplit.v(ptr @fwdsplit_ra_dyn, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %dst, ptr %ddst, metadata !"enzyme_dup", ptr %src, ptr %dsrc, ptr null)
  ret void
}

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Float@double"}

; CHECK: define internal double @fwddiffefwdsplit_const(ptr nocapture %dst, ptr nocapture %"dst'", ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"dst'", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"v'ipl" = load double, ptr %"dst'", align 8
; CHECK-NEXT:   ret double %"v'ipl"

; CHECK: define internal void @fwddiffefwdsplit_ra_dyn(ptr nocapture %dst, ptr nocapture %"dst'", ptr nocapture readonly %src, ptr nocapture %"src'", ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[din:.+]] = icmp eq ptr %"dst'", %dst
; CHECK-NEXT:   %[[sin:.+]] = icmp eq ptr %"src'", %src
; CHECK-NEXT:   %[[skip:.+]] = or i1 %[[din]], %[[sin]]
; CHECK-NEXT:   %[[clen:.+]] = select i1 %[[skip]], i64 0, i64 8
; CHECK-NEXT:   %[[dact:.+]] = xor i1 %[[din]], true
; CHECK-NEXT:   %[[doz:.+]] = and i1 %[[dact]], %[[sin]]
; CHECK-NEXT:   %[[zlen:.+]] = select i1 %[[doz]], i64 8, i64 0
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %"dst'", ptr align 8 %"src'", i64 %[[clen]], i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"dst'", i8 0, i64 %[[zlen]], i1 false)
; CHECK-NEXT:   ret void
