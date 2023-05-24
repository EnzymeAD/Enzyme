; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

define double @tester(double %x, double %y) {
entry:
  %cstx = bitcast double %x to i64 
  %csty = bitcast double %y to i64 
  %andy = and i64 %csty, -9223372036854775808
  %negx = xor i64 %cstx, %andy
  %cstz = bitcast i64 %negx to double
  ret double %cstz
}

define double @test_derivative(double %x, double %dx, double %y, double %dy) {
entry:
  %0 = tail call double (double (double,double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double %dx, double %y, double %dy)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal double @fwddiffetester(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %csty = bitcast double %y to i64
; CHECK-NEXT:   %andy = and i64 %csty, -9223372036854775808
; CHECK-NEXT:   %0 = icmp eq i64 %andy, 0
; CHECK-NEXT:   %1 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %"x'"
; CHECK-NEXT:   %2 = select{{( fast)?}} i1 %0, double %"x'", double %1
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }
