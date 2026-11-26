; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

declare i64 @opaque_dim()
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

define i1 @callee() {
entry:
  %a = call i64 @opaque_dim()
  %b = call i64 @opaque_dim()
  %s = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %a, i64 %b)
  %s_ov = extractvalue { i64, i1 } %s, 1
  %c = call i64 @opaque_dim()
  %u = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %b, i64 %c)
  %u_ov = extractvalue { i64, i1 } %u, 1
  %res = or i1 %s_ov, %u_ov
  ret i1 %res
}

; CHECK: entry
; CHECK-NEXT:   %a = call i64 @opaque_dim(): {[-1]:Integer}
; CHECK-NEXT:   %b = call i64 @opaque_dim(): {[-1]:Integer}
; CHECK-NEXT:   %s = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %a, i64 %b): {[-1]:Integer}
; CHECK-NEXT:   %s_ov = extractvalue { i64, i1 } %s, 1: {[-1]:Integer}
; CHECK-NEXT:   %c = call i64 @opaque_dim(): {[-1]:Integer}
; CHECK-NEXT:   %u = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %b, i64 %c): {[-1]:Integer}
; CHECK-NEXT:   %u_ov = extractvalue { i64, i1 } %u, 1: {[-1]:Integer}
; CHECK-NEXT:   %res = or i1 %s_ov, %u_ov: {[-1]:Integer}
; CHECK-NEXT:   ret i1 %res: {}
