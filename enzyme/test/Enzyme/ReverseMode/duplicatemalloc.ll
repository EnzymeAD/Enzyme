; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -dse -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,inline,dse"  -enzyme-preopt=false -S | FileCheck %s

; #include <stdlib.h>
; #include <stdio.h>
;
; __attribute__((noinline))
; double f(double* x) {
;     return x[0];
; }
;
; double malloced(double x, unsigned long n) {
;     double *array = malloc(sizeof(double)*n);
;     array[0] = x;
;     double res = f(array);
;     free(array);
;     return res * res;
; }
;
; double derivative(double x, unsigned long n) {
;     return __builtin_autodiff(malloced, x, n);
; }
;
; int main(int argc, char** argv) {
;     double x = atof(argv[1]);
;     int n = atoi(argv[2]);
;     printf("original =%f derivative=%f\n", malloced(x, n), derivative(x, n));
; }

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @f(double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  ret double %0
}

; Function Attrs: nounwind uwtable
define dso_local double @malloced(double %x, i64 %n) #1 {
entry:
  %mul = shl i64 %n, 3
  %call = tail call i8* @malloc(i64 %mul)
  %0 = bitcast i8* %call to double*
  store double %x, double* %0, align 8, !tbaa !2
  %call1 = tail call fast double @f(double* %0)
  %call2 = tail call i32 (double*, ...) bitcast (i32 (...)* @free to i32 (double*, ...)*)(double* %0) #4
  %mul3 = fmul fast double %call1, %call1
  ret double %mul3
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

declare dso_local i32 @free(...) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local void @derivative(double %x, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @malloced, double %x, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i64)*, ...) #4

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define dso_local void @derivative(double %x, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"mul3'de.i" = alloca double, align 8
; CHECK-NEXT:   %"call1'de.i" = alloca double, align 8
; CHECK-NEXT:   %"x'de.i" = alloca double, align 8
; CHECK-NEXT:   %0 = bitcast double* %"mul3'de.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %0)
; CHECK-NEXT:   %1 = bitcast double* %"call1'de.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %1)
; CHECK-NEXT:   %2 = bitcast double* %"x'de.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %2)
; CHECK-NEXT:   store double 0.000000e+00, double* %"call1'de.i", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'de.i", align 8
; CHECK-NEXT:   %mul.i = shl i64 %n, 3
; CHECK-NEXT:   %call.i = call i8* @malloc(i64 %mul.i) #8
; CHECK-NEXT:   %"call'mi.i" = call noalias nonnull i8* @malloc(i64 %mul.i) #8
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"call'mi.i", i8 0, i64 %mul.i, i1 false) #4
; CHECK-NEXT:   %"'ipc.i" = bitcast i8* %"call'mi.i" to double*
; CHECK-NEXT:   %3 = bitcast i8* %call.i to double*
; CHECK-NEXT:   store double %x, double* %3, align 8, !tbaa !2
; CHECK-NEXT:   %call1.i = call fast double @augmented_f(double* %3, double* %"'ipc.i") #4
; CHECK-NEXT:   store double 1.000000e+00, double* %"mul3'de.i", align 8
; CHECK-NEXT:   %4 = load double, double* %"mul3'de.i", align 8
; CHECK-NEXT:   %m0diffecall1.i = fmul fast double %4, %call1.i
; CHECK-NEXT:   %m1diffecall1.i = fmul fast double %4, %call1.i
; CHECK-NEXT:   %5 = load double, double* %"call1'de.i", align 8
; CHECK-NEXT:   %6 = fadd fast double %5, %m0diffecall1.i
; CHECK-NEXT:   store double %6, double* %"call1'de.i", align 8
; CHECK-NEXT:   %7 = load double, double* %"call1'de.i", align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %m1diffecall1.i
; CHECK-NEXT:   store double %8, double* %"call1'de.i", align 8
; CHECK-NEXT:   %9 = load double, double* %"call1'de.i", align 8
; CHECK-NEXT:   call void @diffef(double* %3, double* %"'ipc.i", double %9) #4
; CHECK-NEXT:   %10 = load double, double* %"'ipc.i", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'ipc.i", align 8
; CHECK-NEXT:   %11 = load double, double* %"x'de.i", align 8
; CHECK-NEXT:   %12 = fadd fast double %11, %10
; CHECK-NEXT:   store double %12, double* %"x'de.i", align 8
; CHECK-NEXT:   call void bitcast (i32 (...)* @free to void (i8*)*)(i8* nonnull %"call'mi.i") #4
; CHECK-NEXT:   call void bitcast (i32 (...)* @free to void (i8*)*)(i8* %call.i) #4
; CHECK-NEXT:   %13 = load double, double* %"x'de.i", align 8
; CHECK-NEXT:   %14 = insertvalue { double } undef, double %13, 0
; CHECK-NEXT:   %15 = bitcast double* %"mul3'de.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %15)
; CHECK-NEXT:   %16 = bitcast double* %"call1'de.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %16)
; CHECK-NEXT:   %17 = bitcast double* %"x'de.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %17)
; CHECK-NEXT:   %18 = extractvalue { double } %14, 0
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffef(double* nocapture readonly %x, double* nocapture %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'de" = alloca double, align 8
; CHECK-NEXT:   br label %invertentry

; CHECK:  invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double %differeturn, double* %"'de", align 8
; CHECK-NEXT:   %0 = load double, double* %"'de", align 8
; CHECK-NEXT:   %1 = load double, double* %"x'", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   store double %2, double* %"x'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
