; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %cstx = bitcast double %x to i64 
  %csty = bitcast double %y to i64 
  %andy = and i64 %csty, -9223372036854775808
  %negx = xor i64 %cstx, %andy
  %cstz = bitcast i64 %negx to double
  ret double %cstz
}

define { double, double } @test_derivative(double %x, double %y) {
entry:
  %0 = tail call { double, double } (...) @__enzyme_autodiff(double (double, double)* @tester, double %x, double %y)
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_autodiff(...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %csty = bitcast double %y to i64
; CHECK-NEXT:   %andy = and i64 %csty, -9223372036854775808
; CHECK-NEXT:   %0 = fadd double 0.000000e+00, %differeturn
; CHECK-NEXT:   %1 = icmp eq i64 %andy, 0
; CHECK-NEXT:   %2 = {{(fsub double \-?0.000000e\+00,|fneg double)}} %0
; CHECK-NEXT:   %3 = select{{( fast)?}} i1 %1, double %0, double %2
; CHECK-NEXT:   %4 = fadd double 0.000000e+00, %3
; CHECK-NEXT:   %5 = insertvalue { double, double } undef, double %4, 0
; CHECK-NEXT:   %6 = insertvalue { double, double } %5, double 0.000000e+00, 1
; CHECK-NEXT:   ret { double, double } %6
; CHECK-NEXT: }
