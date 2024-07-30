; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.cos.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_error_estimate(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_error_estimate(double (double)*, ...)

; Function Attrs: mustprogress noinline optnone ssp uwtable
declare void @enzymeLogValue(i8* noundef %id, double noundef %res, i32 noundef %numOperands, double* noundef %operands)

; Function Attrs: mustprogress noinline optnone ssp uwtable
declare void @enzymeLogError(i8* noundef %id, double noundef %err)


; CHECK: define internal double @fwderrtester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = alloca [1 x double], align 8
; CHECK-NEXT:   %[[i1:.+]] = tail call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %[[i2:.+]] = getelementptr [1 x double], [1 x double]* %[[i0]], i32 0, i32 0
; CHECK-NEXT:   store double %x, double* %[[i2]], align 8
; CHECK-NEXT:   %[[i3:.+]] = getelementptr [1 x double], [1 x double]* %[[i0]], i32 0, i32 0
; CHECK-NEXT:   call void @enzymeLogValue(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @0, i32 0, i32 0), double %1, i32 1, double* %[[i3]])
; CHECK-NEXT:   %[[i4:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[i5:.+]] = fdiv fast double %[[i4]], %[[i1]]
; CHECK-NEXT:   %[[i6:.+]] = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %[[i7:.+]] = fneg fast double %[[i6]]
; CHECK-NEXT:   %[[i8:.+]] = fmul fast double %[[i5]], %[[i7]]
; CHECK-NEXT:   %[[i9:.+]] = call fast double @llvm.fabs.f64(double %[[i8]])
; CHECK-NEXT:   %[[i10:.+]] = bitcast double %[[i1]] to i64
; CHECK-NEXT:   %[[i11:.+]] = xor i64 %[[i10]], 1
; CHECK-NEXT:   %[[i12:.+]] = bitcast i64 %[[i11]] to double
; CHECK-NEXT:   %[[i13:.+]] = fsub fast double %[[i1]], %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = call fast double @llvm.fabs.f64(double %[[i13]])
; CHECK-NEXT:   %[[i15:.+]] = call fast double @llvm.maxnum.f64(double %[[i14]], double %[[i9]])
; CHECK-NEXT:   call void @enzymeLogError(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @1, i32 0, i32 0), double %[[i15]])
; CHECK-NEXT:   ret double %[[i15]]
; CHECK-NEXT: }
