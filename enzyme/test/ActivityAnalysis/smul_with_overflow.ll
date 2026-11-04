; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=callee -S -o /dev/null | FileCheck %s

declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

define double @callee(double %x, i64 %n, i64 %m) {
entry:
  %s = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %n, i64 %m)
  %s_val = extractvalue { i64, i1 } %s, 0
  %u = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %s_val, i64 8)
  %u_ov = extractvalue { i64, i1 } %u, 1
  %sel = select i1 %u_ov, double 0.000000e+00, double %x
  ret double %sel
}

; CHECK: entry
; CHECK-NEXT:   %s = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %n, i64 %m): icv:1 ici:1
; CHECK-NEXT:   %s_val = extractvalue { i64, i1 } %s, 0: icv:1 ici:1
; CHECK-NEXT:   %u = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %s_val, i64 8): icv:1 ici:1
; CHECK-NEXT:   %u_ov = extractvalue { i64, i1 } %u, 1: icv:1 ici:1
; CHECK-NEXT:   %sel = select i1 %u_ov, double 0.000000e+00, double %x: icv:0 ici:0
