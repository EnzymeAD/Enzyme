; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; Function Attrs: mustprogress noinline optnone ssp uwtable
declare void @enzymeLogValue(i8* noundef %id, double noundef %res, i32 noundef %numOperands, double* noundef %operands)

; Function Attrs: mustprogress noinline optnone ssp uwtable
declare void @enzymeLogGrad(i8* noundef %id, double noundef %grad)


; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[i1:.+]] = fadd fast double %x, %y
; CHECK-NEXT:   %[[i2:.+]] = getelementptr [2 x double], [2 x double]* %[[i0]], i32 0, i32 0
; CHECK-NEXT:   store double %x, double* %[[i2]], align 8
; CHECK-NEXT:   %[[i3:.+]] = getelementptr [2 x double], [2 x double]* %[[i0]], i32 0, i32 1
; CHECK-NEXT:   store double %y, double* %[[i3]], align 8
; CHECK-NEXT:   %[[i4:.+]] = getelementptr [2 x double], [2 x double]* %[[i0]], i32 0, i32 0
; CHECK-NEXT:   call void @enzymeLogValue(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @0, i32 0, i32 0), double %[[i1]], i32 2, double* %[[i4]])
; CHECK-NEXT:   call void @enzymeLogGrad(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @1, i32 0, i32 0), double %[[differet]])
; CHECK-NEXT:   %[[i5:.+]] = insertvalue { double, double } undef, double %[[differet]], 0
; CHECK-NEXT:   %[[i6:.+]] = insertvalue { double, double } %[[i5]], double %[[differet]], 1
; CHECK-NEXT:   ret { double, double } %[[i6]]
; CHECK-NEXT: }
