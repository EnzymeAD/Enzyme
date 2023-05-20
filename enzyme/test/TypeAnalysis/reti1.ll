; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=d -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=d -S -o /dev/null | FileCheck %s

define dso_local double @m(double %div) {
entry:
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @_Z6foobard to i8*), double %div) #3
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double) local_unnamed_addr #0

; Function Attrs: nounwind uwtable
define dso_local double @_Z6foobard(double %t) #1 {
entry:
  %alloc = alloca i64
  store i64 ptrtoint (i1 (double*)* @d to i64), i64* %alloc
  ret double 1.000000e-02
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local zeroext i1 @d(double* %x) #2 align 2 {
entry:
  %call = tail call zeroext i1 @g(double* %x)
  ret i1 %call
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local zeroext i1 @g(double* %x) local_unnamed_addr #2 {
entry:
  ret i1 false
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" }
attributes #1 = { nounwind uwtable }
attributes #2 = { norecurse nounwind uwtable }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}

; CHECK: d - {[-1]:Integer} |{[-1]:Pointer, [-1,-1]:Float@double}:{} 
; CHECK-NEXT: double* %x: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %call = tail call zeroext i1 @g(double* %x): {[-1]:Integer}
; CHECK-NEXT:   ret i1 %call: {}

; CHECK-NOT: g
