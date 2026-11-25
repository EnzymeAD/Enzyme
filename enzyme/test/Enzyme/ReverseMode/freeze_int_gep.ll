; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

declare void @__enzyme_autodiff(...)
declare double @__enzyme_autodiff_d(...)

define void @f_gep(ptr %0, i64 %1, ptr %2, ptr %3) {
entry:
  %4 = load i64, ptr %2, align 8
  %5 = freeze i64 %4
  %6 = icmp eq i64 %1, 0
  br i1 %6, label %common.ret1, label %7

common.ret1:
  ret void

7:
  %8 = phi i64 [ 0, %entry ]
  %9 = add i64 %8, %5
  %10 = getelementptr inbounds double, ptr %3, i64 %9
  %11 = load double, ptr %0, align 8
  %12 = load double, ptr %10, align 8
  %13 = fmul double %11, %12
  store double %13, ptr %0, align 8
  br label %common.ret1
}

define double @f_bitcast_freeze(double %x) {
entry:
  %bc1 = bitcast double %x to i64
  %fr = freeze i64 %bc1
  %bc2 = bitcast i64 %fr to double
  %res = fmul double %bc2, %bc2
  ret double %res
}

define double @caller(ptr %0, ptr %d0, i64 %1, ptr %2, ptr %d2, ptr %3, ptr %d3, double %x) {
  call void (...) @__enzyme_autodiff(ptr @f_gep, metadata !"enzyme_dup", ptr %0, ptr %d0, metadata !"enzyme_const", i64 %1, metadata !"enzyme_dup", ptr %2, ptr %d2, metadata !"enzyme_dup", ptr %3, ptr %d3)
  %dx = call double (...) @__enzyme_autodiff_d(ptr @f_bitcast_freeze, double %x)
  ret double %dx
}

; CHECK: define internal void @diffef_gep(
; CHECK:   %[[ld:.+]] = load i64, ptr %2, align 8
; CHECK-NEXT:   %[[fr:.+]] = freeze i64 %[[ld]]
; CHECK:   %[[ipg:.+]] = getelementptr inbounds double, ptr %"'2", i64 %[[fr]]

; CHECK: define internal { double } @diffef_bitcast_freeze(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %bc1 = bitcast double %x to i64
; CHECK-NEXT:   %fr = freeze i64 %bc1
; CHECK-NEXT:   %bc2 = bitcast i64 %fr to double
; CHECK-NEXT:   %0 = fmul fast double %differeturn, %bc2
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %bc2
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   %3 = bitcast double %2 to i64
; CHECK-NEXT:   %4 = freeze i64 %3
; CHECK-NEXT:   %5 = bitcast i64 %4 to double
; CHECK-NEXT:   %6 = insertvalue { double } undef, double %5, 0
; CHECK-NEXT:   ret { double } %6
; CHECK-NEXT: }
