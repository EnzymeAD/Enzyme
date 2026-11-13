; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=step_32bit -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=step_32bit -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n8:16:32-S128"

define void @step_32bit(ptr %p32, ptr %p16) {
entry:
  %v32 = load i32, ptr %p32, align 4
  %inc32 = add i32 %v32, 1
  store i32 %inc32, ptr %p32, align 4
  %v16 = load i16, ptr %p16, align 2
  %inc16 = add i16 %v16, 1
  store i16 %inc16, ptr %p16, align 2
  ret void
}

; On a 32-bit pointer target, i32 can hold a pointer so unknown-type add i32
; must remain active, whereas i16 is narrower than a pointer and is inactive.
; CHECK: entry
; CHECK-NEXT:   %v32 = load i32, ptr %p32, align 4: icv:0 ici:0
; CHECK-NEXT:   %inc32 = add i32 %v32, 1: icv:0 ici:0
; CHECK-NEXT:   store i32 %inc32, ptr %p32, align 4: icv:1 ici:0
; CHECK-NEXT:   %v16 = load i16, ptr %p16, align 2: icv:1 ici:1
; CHECK-NEXT:   %inc16 = add i16 %v16, 1: icv:1 ici:1
; CHECK-NEXT:   store i16 %inc16, ptr %p16, align 2: icv:1 ici:1
