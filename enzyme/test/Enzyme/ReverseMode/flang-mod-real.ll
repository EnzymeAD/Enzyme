; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg,adce)" -S | FileCheck %s; fi

; LLVM flang calls its runtime for MOD and MODULO of reals,
; _FortranAModReal8(a, p, sourceFile, sourceLine). d/da = 1,
; d/dp = -trunc(a/p) for MOD and -floor(a/p) for MODULO.

@file = private constant [6 x i8] c"m.f90\00"

define double @mod(double %a, double %p) {
entry:
  %r = call double @_FortranAModReal8(double %a, double %p, ptr @file, i32 3)
  ret double %r
}

define double @modulo(double %a, double %p) {
entry:
  %r = call double @_FortranAModuloReal8(double %a, double %p, ptr @file, i32 7)
  ret double %r
}

define void @test_derivative(double %a, double %p) {
entry:
  %0 = call { double, double } (...) @__enzyme_autodiff(ptr @mod, double %a, double %p)
  %1 = call { double, double } (...) @__enzyme_autodiff(ptr @modulo, double %a, double %p)
  ret void
}

declare double @_FortranAModReal8(double, double, ptr, i32)
declare double @_FortranAModuloReal8(double, double, ptr, i32)
declare { double, double } @__enzyme_autodiff(...)

; CHECK: define internal { double, double } @diffemod(double %a, double %p, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[q:.+]] = fdiv fast double %a, %p
; CHECK-NEXT:   %[[t:.+]] = call fast double @llvm.trunc.f64(double %[[q]])
; CHECK-NEXT:   %[[n:.+]] = fneg fast double %[[t]]
; CHECK-NEXT:   %[[dp:.+]] = fmul fast double %differeturn, %[[n]]
; CHECK-NEXT:   %[[r0:.+]] = insertvalue { double, double } undef, double %differeturn, 0
; CHECK-NEXT:   %[[r1:.+]] = insertvalue { double, double } %[[r0]], double %[[dp]], 1
; CHECK-NEXT:   ret { double, double } %[[r1]]

; CHECK: define internal { double, double } @diffemodulo(double %a, double %p, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[q:.+]] = fdiv fast double %a, %p
; CHECK-NEXT:   %[[f:.+]] = call fast double @llvm.floor.f64(double %[[q]])
; CHECK-NEXT:   %[[n:.+]] = fneg fast double %[[f]]
; CHECK-NEXT:   %[[dp:.+]] = fmul fast double %differeturn, %[[n]]
; CHECK-NEXT:   %[[r0:.+]] = insertvalue { double, double } undef, double %differeturn, 0
; CHECK-NEXT:   %[[r1:.+]] = insertvalue { double, double } %[[r0]], double %[[dp]], 1
; CHECK-NEXT:   ret { double, double } %[[r1]]
