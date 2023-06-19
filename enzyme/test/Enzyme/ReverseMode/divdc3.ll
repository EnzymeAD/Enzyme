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
; CHECK-NEXT:   %0 = fmul fast double %re1, %yre
; CHECK-NEXT:   %1 = fmul fast double %im1, %yim
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   %3 = fmul fast double %yre, %yre
; CHECK-NEXT:   %4 = fmul fast double %yim, %yim
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   %6 = fdiv fast double %2, %5
; CHECK-NEXT:   %7 = fmul fast double %yre, %im1
; CHECK-NEXT:   %8 = fmul fast double %re1, %yim
; CHECK-NEXT:   %9 = fsub fast double %7, %8
; CHECK-NEXT:   %10 = fdiv fast double %9, %5
; CHECK-NEXT:   %11 = fadd fast double %6, %10
; CHECK-NEXT:   %12 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %yre
; CHECK-NEXT:   %13 = fmul fast double %im1, %12
; CHECK-NEXT:   %14 = fadd fast double %8, %13
; CHECK-NEXT:   %15 = fmul fast double %12, %12
; CHECK-NEXT:   %16 = fadd fast double %4, %15
; CHECK-NEXT:   %17 = fdiv fast double %14, %16
; CHECK-NEXT:   %18 = fmul fast double %re1, %12
; CHECK-NEXT:   %19 = fsub fast double %1, %18
; CHECK-NEXT:   %20 = fdiv fast double %19, %16
; CHECK-NEXT:   %21 = fadd fast double %17, %20
; CHECK-NEXT:   %22 = fmul fast double %re1, %xre
; CHECK-NEXT:   %23 = fmul fast double %im1, %xim
; CHECK-NEXT:   %24 = fsub fast double %22, %23
; CHECK-NEXT:   %25 = fmul fast double %re1, %xim
; CHECK-NEXT:   %26 = fmul fast double %xre, %im1
; CHECK-NEXT:   %27 = fadd fast double %25, %26
; CHECK-NEXT:   %28 = fsub fast double %3, %4
; CHECK-NEXT:   %29 = fmul fast double %yre, %yim
; CHECK-NEXT:   %30 = fadd fast double %29, %29
; CHECK-NEXT:   %31 = fmul fast double %24, %28
; CHECK-NEXT:   %32 = fmul fast double %27, %30
; CHECK-NEXT:   %33 = fadd fast double %31, %32
; CHECK-NEXT:   %34 = fmul fast double %28, %28
; CHECK-NEXT:   %35 = fmul fast double %30, %30
; CHECK-NEXT:   %36 = fadd fast double %34, %35
; CHECK-NEXT:   %37 = fdiv fast double %33, %36
; CHECK-NEXT:   %38 = fmul fast double %28, %27
; CHECK-NEXT:   %39 = fmul fast double %24, %30
; CHECK-NEXT:   %40 = fsub fast double %38, %39
; CHECK-NEXT:   %41 = fdiv fast double %40, %36
; CHECK-NEXT:   %42 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %37
; CHECK-NEXT:   %43 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %41
; CHECK-NEXT:   %44 = fadd fast double %42, %43
; CHECK-NEXT:   %45 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %xim
; CHECK-NEXT:   %46 = fmul fast double %im1, %45
; CHECK-NEXT:   %47 = fsub fast double %25, %46
; CHECK-NEXT:   %48 = fmul fast double %re1, %45
; CHECK-NEXT:   %49 = fadd fast double %48, %23
; CHECK-NEXT:   %50 = fmul fast double %47, %28
; CHECK-NEXT:   %51 = fmul fast double %49, %30
; CHECK-NEXT:   %52 = fadd fast double %50, %51
; CHECK-NEXT:   %53 = fdiv fast double %52, %36
; CHECK-NEXT:   %54 = fmul fast double %28, %49
; CHECK-NEXT:   %55 = fmul fast double %47, %30
; CHECK-NEXT:   %56 = fsub fast double %54, %55
; CHECK-NEXT:   %57 = fdiv fast double %56, %36
; CHECK-NEXT:   %58 = fadd fast double %53, %57
; CHECK-NEXT:   %59 = insertvalue { double, double, double, double } {{(undef|poison)?}}, double %11, 0
; CHECK-NEXT:   %60 = insertvalue { double, double, double, double } %59, double %21, 1
; CHECK-NEXT:   %61 = insertvalue { double, double, double, double } %60, double %44, 2
; CHECK-NEXT:   %62 = insertvalue { double, double, double, double } %61, double %58, 3
; CHECK-NEXT:   ret { double, double, double, double } %62
; CHECK-NEXT: }
