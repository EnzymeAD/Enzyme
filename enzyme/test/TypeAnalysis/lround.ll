; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -ge 10 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s; fi

declare double @f()
declare i32 @llvm.lround.i32.f64(double)
declare i64 @llvm.llround.i64.f64(double)
declare i64 @llvm.lrint.i64.f64(double)
declare i64 @llvm.llrint.i64.f64(double)

; The result of rounding a float to an integer is an integer, so a value it
; is added to is one too.
define void @caller() {
entry:
  %c = call double @f()
  %r = call i32 @llvm.lround.i32.f64(double %c)
  %s = add i32 %r, %r
  %d = call double @f()
  %ll = call i64 @llvm.llround.i64.f64(double %d)
  %e = call double @f()
  %l = call i64 @llvm.lrint.i64.f64(double %e)
  %g = call double @f()
  %lll = call i64 @llvm.llrint.i64.f64(double %g)
  ret void
}

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %c = call double @f(): {[-1]:Float@double}
; CHECK-NEXT:   %r = call i32 @llvm.lround.i32.f64(double %c): {[-1]:Integer}
; CHECK-NEXT:   %s = add i32 %r, %r: {[-1]:Integer}
; CHECK-NEXT:   %d = call double @f(): {[-1]:Float@double}
; CHECK-NEXT:   %ll = call i64 @llvm.llround.i64.f64(double %d): {[-1]:Integer}
; CHECK-NEXT:   %e = call double @f(): {[-1]:Float@double}
; CHECK-NEXT:   %l = call i64 @llvm.lrint.i64.f64(double %e): {[-1]:Integer}
; CHECK-NEXT:   %g = call double @f(): {[-1]:Float@double}
; CHECK-NEXT:   %lll = call i64 @llvm.llrint.i64.f64(double %g): {[-1]:Integer}
; CHECK-NEXT:   ret void: {}
