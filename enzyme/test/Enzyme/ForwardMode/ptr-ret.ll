; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define dso_local noalias nonnull double* @_Z6toHeapd(double %x) local_unnamed_addr #0 {
entry:
  %call = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8) #4
  %0 = bitcast i8* %call to double*
  store double %x, double* %0, align 8
  ret double* %0
}

declare dso_local nonnull i8* @_Znwm(i64) local_unnamed_addr #1

define dso_local double @_Z6squared(double %x) #0 {
entry:
  %call = call double* @_Z6toHeapd(double %x)
  %0 = load double, double* %call, align 8
  %mul = fmul double %0, %x
  ret double %mul
}

define dso_local double @_Z7dsquared(double %x) local_unnamed_addr #2 {
entry:
  %call = call double (...) @_Z16__enzyme_fwddiffz(i8* bitcast (double (double)* @_Z6squared to i8*), double %x, double 1.000000e+00)
  ret double %call
}

declare dso_local double @_Z16__enzyme_fwddiffz(...) local_unnamed_addr #3

attributes #0 = { nofree uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nobuiltin nofree allocsize(0) "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { builtin allocsize(0) }




; CHECK: define dso_local double @_Z7dsquared(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @fwddiffe_Z6squared(double %x, double 1.000000e+00)
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffe_Z6squared(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call double* @_Z6toHeapd(double %x)
; CHECK-NEXT:   %0 = call { double*, double* } @fwddiffe_Z6toHeapd(double %x, double %"x'")
; CHECK-NEXT:   %1 = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   %2 = load double, double* %call, align 8
; CHECK-NEXT:   %3 = load double, double* %1, align 8
; CHECK-NEXT:   %4 = fmul fast double %3, %x
; CHECK-NEXT:   %5 = fmul fast double %"x'", %2
; CHECK-NEXT:   %6 = fadd fast double %4, %5
; CHECK-NEXT:   ret double %6
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @fwddiffe_Z6toHeapd(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8) #4
; CHECK-NEXT:   %0 = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8) #4
; CHECK-NEXT:   %"'ipc" = bitcast i8* %0 to double*
; CHECK-NEXT:   %1 = bitcast i8* %call to double*
; CHECK-NEXT:   store double %x, double* %1, align 8
; CHECK-NEXT:   store double %"x'", double* %"'ipc", align 8
; CHECK-NEXT:   %2 = insertvalue { double*, double* } undef, double* %1, 0
; CHECK-NEXT:   %3 = insertvalue { double*, double* } %2, double* %"'ipc", 1
; CHECK-NEXT:   ret { double*, double* } %3
; CHECK-NEXT: }