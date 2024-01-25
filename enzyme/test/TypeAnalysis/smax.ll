; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=smax -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=smax -S -o /dev/null | FileCheck %s

define i32 @smax(i32 %a, i32 %b) {
entry:
  %0 = call i32 @llvm.smax.i32(i32 %a, i32 %b)
  %1 = add i32 %a, 1
  %2 = add i32 %b, 2
  %3 = call i32 @llvm.smax.i32(i32 %1, i32 %2)
  ret i32 %3
}

declare i32 @llvm.smax.i32(i32, i32)


; CHECK: smax - smax - {[-1]:Integer} |{[-1]:Integer}:{} {[-1]:Integer}:{} 
; CHECK-NEXT: i32 %a: {[-1]:Integer}
; CHECK_NEXT: i32 %b: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %0 = call i32 @llvm.smax.i32(i32 %a, i32 %b): {}
; CHECK-NEXT:   %1 = add i32 %a, 1: {[-1]:Integer}
; CHECK-NEXT:   %2 = add i32 %b, 2: {[-1]:Integer}
; CHECK-NEXT:   %3 = call i32 @llvm.smax.i32(i32 %1, i32 %2): {[-1]:Integer}
; CHECK-NEXT:   ret i32 %3: {}