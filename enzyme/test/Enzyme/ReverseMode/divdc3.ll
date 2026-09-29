; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false  -enzyme -mem2reg -simplifycfg -early-cse -instsimplify  -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,early-cse,instsimplify)" -S | FileCheck %s

define { double, double } @test(double %xre, double %xim, double %yre, double %yim) {
entry:
  %call = call { double, double } @__divdc3(double %xre, double %xim, double %yre, double %yim)
  ret { double, double } %call
}

declare { double, double } @__divdc3(double, double, double, double)

define { double, double, double, double} @dtest(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double, double, double } (i8*, ...) @__enzyme_autodiff(i8* bitcast ({ double, double } (double, double, double, double)* @test to i8*), double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
  ret { double, double, double, double} %call
}

declare { double, double, double, double } @__enzyme_autodiff(i8*, ...)


; CHECK: define internal { double, double, double, double } @diffetest(double %xre, double %xim, double %yre, double %yim, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %re1 = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %im1 = extractvalue { double, double } %differeturn, 1
; CHECK-NEXT:   %0 = fmul double %re1, %yre
; CHECK-NEXT:   %1 = fmul double %im1, %yim
; CHECK-NEXT:   %2 = fadd double %0, %1
; CHECK-NEXT:   %3 = fmul double %yre, %yre
; CHECK-NEXT:   %4 = fmul double %yim, %yim
; CHECK-NEXT:   %5 = fadd double %3, %4
; CHECK-NEXT:   %6 = fdiv double %2, %5
; CHECK-NEXT:   %7 = fmul double %yre, %im1
; CHECK-NEXT:   %8 = fmul double %re1, %yim
; CHECK-NEXT:   %9 = fsub double %7, %8
; CHECK-NEXT:   %10 = fdiv double %9, %5
; CHECK-NEXT:   %11 = fadd double %6, %10
; CHECK-NEXT:   %12 = fadd double 0.000000e+00, %11
; CHECK-NEXT:   %13 = {{(fneg double)|(fsub double \-0.000000e\+00,)}} %yre
; CHECK-NEXT:   %14 = fmul double %im1, %13
; CHECK-NEXT:   %15 = fadd double %8, %14
; CHECK-NEXT:   %16 = fmul double %13, %13
; CHECK-NEXT:   %17 = fadd double %4, %16
; CHECK-NEXT:   %18 = fdiv double %15, %17
; CHECK-NEXT:   %19 = fmul double %re1, %13
; CHECK-NEXT:   %20 = fsub double %1, %19
; CHECK-NEXT:   %21 = fdiv double %20, %17
; CHECK-NEXT:   %22 = fadd double %18, %21
; CHECK-NEXT:   %23 = fadd double 0.000000e+00, %22
; CHECK-NEXT:   %24 = fmul double %re1, %xre
; CHECK-NEXT:   %25 = fmul double %im1, %xim
; CHECK-NEXT:   %26 = fsub double %24, %25
; CHECK-NEXT:   %27 = fmul double %re1, %xim
; CHECK-NEXT:   %28 = fmul double %xre, %im1
; CHECK-NEXT:   %29 = fadd double %27, %28
; CHECK-NEXT:   %30 = fsub double %3, %4
; CHECK-NEXT:   %31 = fmul double %yre, %yim
; CHECK-NEXT:   %32 = fadd double %31, %31
; CHECK-NEXT:   %33 = fmul double %26, %30
; CHECK-NEXT:   %34 = fmul double %29, %32
; CHECK-NEXT:   %35 = fadd double %33, %34
; CHECK-NEXT:   %36 = fmul double %30, %30
; CHECK-NEXT:   %37 = fmul double %32, %32
; CHECK-NEXT:   %38 = fadd double %36, %37
; CHECK-NEXT:   %39 = fdiv double %35, %38
; CHECK-NEXT:   %40 = fmul double %30, %29
; CHECK-NEXT:   %41 = fmul double %26, %32
; CHECK-NEXT:   %42 = fsub double %40, %41
; CHECK-NEXT:   %43 = fdiv double %42, %38
; CHECK-NEXT:   %44 = {{(fneg double)|(fsub double \-0.000000e\+00,)}} %39
; CHECK-NEXT:   %45 = {{(fneg double)|(fsub double \-0.000000e\+00,)}} %43
; CHECK-NEXT:   %46 = fadd double %44, %45
; CHECK-NEXT:   %47 = fadd double 0.000000e+00, %46
; CHECK-NEXT:   %48 = {{(fneg double)|(fsub double \-0.000000e\+00,)}} %xim
; CHECK-NEXT:   %49 = fmul double %im1, %48
; CHECK-NEXT:   %50 = fsub double %27, %49
; CHECK-NEXT:   %51 = fmul double %re1, %48
; CHECK-NEXT:   %52 = fadd double %51, %25
; CHECK-NEXT:   %53 = fmul double %50, %30
; CHECK-NEXT:   %54 = fmul double %52, %32
; CHECK-NEXT:   %55 = fadd double %53, %54
; CHECK-NEXT:   %56 = fdiv double %55, %38
; CHECK-NEXT:   %57 = fmul double %30, %52
; CHECK-NEXT:   %58 = fmul double %50, %32
; CHECK-NEXT:   %59 = fsub double %57, %58
; CHECK-NEXT:   %60 = fdiv double %59, %38
; CHECK-NEXT:   %61 = fadd double %56, %60
; CHECK-NEXT:   %62 = fadd double 0.000000e+00, %61
; CHECK-NEXT:   %63 = insertvalue { double, double, double, double } {{(undef|poison)?}}, double %12, 0
; CHECK-NEXT:   %64 = insertvalue { double, double, double, double } %63, double %23, 1
; CHECK-NEXT:   %65 = insertvalue { double, double, double, double } %64, double %47, 2
; CHECK-NEXT:   %66 = insertvalue { double, double, double, double } %65, double %62, 3
; CHECK-NEXT:   ret { double, double, double, double } %66
; CHECK-NEXT: }
