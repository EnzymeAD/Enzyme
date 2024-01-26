; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=smax -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=smax -S -o /dev/null | FileCheck %s

define i32 @smax(i32 %a, i32 %b) {
entry:
  %0 = call i32 @llvm.smax.i32(i32 %a, i32 %b)
  %1 = call i32 @getint()
  %2 = call i32 @getint()
  %3 = add i32 %a, %1
  %4 = add i32 %b, %2
  %5 = call i32 @llvm.smax.i32(i32 %3, i32 %4)
  ret i32 %5
}

declare i32 @llvm.smax.i32(i32, i32)

declare i32 @getint()


; CHECK: smax - {[-1]:Integer} |{[-1]:Integer}:{} {[-1]:Integer}:{}
; CHECK-NEXT: i32 %a: {[-1]:Integer}
; CHECK_NEXT: i32 %b: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT: %1 = call i32 @getint(): {[-1]:Integer}
; CHECK-NEXT: %2 = call i32 @getint(): {[-1]:Integer}
; CHECK-NEXT: %3 = add i32 %a, %1: {[-1]:Integer}
; CHECK-NEXT: %4 = add i32 %b, %2: {[-1]:Integer}
; CHECK-NEXT: %5 = call i32 @llvm.smax.i32(i32 %3, i32 %4): {[-1]:Integer}
; CHECK-NEXT: ret i32 %5: {}