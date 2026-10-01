; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

; An integer and'ed with a constant stays an integer, also when the mask is
; not a small one (e.g. flang masking a CHARACTER length with 0x7fffffff).

define i64 @callee(i64 %a, i64 %b) {
entry:
  %n = sdiv i64 %a, %b
  %m = and i64 %n, 2147483647
  %k = and i64 4294967295, %n
  ret i64 %m
}

; CHECK: callee - {[-1]:Integer} |{[-1]:Integer}:{} {[-1]:Integer}:{}
; CHECK-NEXT: i64 %a: {[-1]:Integer}
; CHECK-NEXT: i64 %b: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %n = sdiv i64 %a, %b: {[-1]:Integer}
; CHECK-NEXT:   %m = and i64 %n, 2147483647: {[-1]:Integer}
; CHECK-NEXT:   %k = and i64 4294967295, %n: {[-1]:Integer}
