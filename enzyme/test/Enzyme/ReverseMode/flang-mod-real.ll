; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="preserve-nvvm,enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg,adce)" -S | FileCheck %s; fi

; LLVM flang lowers MOD and MODULO of reals to calls into its runtime,
; _FortranAModReal8(a, p, sourceFile, sourceLine). preserve-nvvm replaces
; them with frem (plus the sign fix-up for MODULO), which Enzyme
; differentiates: d/da = 1, d/dp = -trunc(a/p) for MOD and
; -trunc(a/p) + 1 = -floor(a/p) for MODULO when the remainder is adjusted.

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

; CHECK: define double @mod(double %a, double %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r = frem double %a, %p
; CHECK-NEXT:   ret double %r

; CHECK: define double @modulo(double %a, double %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[rem:.+]] = frem double %a, %p
; CHECK-NEXT:   %[[pneg:.+]] = fcmp olt double %p, 0.000000e+00
; CHECK-NEXT:   %[[aneg:.+]] = fcmp olt double %a, 0.000000e+00
; CHECK-NEXT:   %[[differ:.+]] = xor i1 %[[aneg]], %[[pneg]]
; CHECK-NEXT:   %[[nz:.+]] = fcmp une double %[[rem]], 0.000000e+00
; CHECK-NEXT:   %[[adj:.+]] = and i1 %[[nz]], %[[differ]]
; CHECK-NEXT:   %[[radd:.+]] = fadd double %[[rem]], %p
; CHECK-NEXT:   %[[sel:.+]] = select i1 %[[adj]], double %[[radd]], double %[[rem]]
; CHECK-NEXT:   %[[pabs:.+]] = call double @llvm.fabs.f64(double %p)
; CHECK-NEXT:   %[[pinf:.+]] = fcmp oeq double %[[pabs]], 0x7FF0000000000000
; CHECK-NEXT:   %r = select i1 %[[pinf]], double 0x7FF8000000000000, double %[[sel]]
; CHECK-NEXT:   ret double %r

; CHECK: define internal { double, double } @diffemod(double %a, double %p, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fdiv fast double %a, %p
; CHECK-NEXT:   %[[i1:.+]] = call fast double @llvm.fabs.f64(double %[[i0]])
; CHECK-NEXT:   %[[i2:.+]] = call fast double @llvm.floor.f64(double %[[i1]])
; CHECK-NEXT:   %[[i3:.+]] = call fast double @llvm.copysign.f64(double %[[i2]], double %[[i0]])
; CHECK-NEXT:   %[[i4:.+]] = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %[[i3]]
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %differeturn, %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = insertvalue { double, double } undef, double %differeturn, 0
; CHECK-NEXT:   %[[i7:.+]] = insertvalue { double, double } %[[i6]], double %[[i5]], 1
; CHECK-NEXT:   ret { double, double } %[[i7]]

; CHECK: define internal { double, double } @diffemodulo(double %a, double %p, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[rem:.+]] = frem double %a, %p
; CHECK-NEXT:   %[[pneg:.+]] = fcmp olt double %p, 0.000000e+00
; CHECK-NEXT:   %[[aneg:.+]] = fcmp olt double %a, 0.000000e+00
; CHECK-NEXT:   %[[differ:.+]] = xor i1 %[[aneg]], %[[pneg]]
; CHECK-NEXT:   %[[nz:.+]] = fcmp une double %[[rem]], 0.000000e+00
; CHECK-NEXT:   %[[adj:.+]] = and i1 %[[nz]], %[[differ]]
; CHECK-NEXT:   %[[pabs:.+]] = call double @llvm.fabs.f64(double %p)
; CHECK-NEXT:   %[[pinf:.+]] = fcmp oeq double %[[pabs]], 0x7FF0000000000000
; CHECK-NEXT:   %[[dsel:.+]] = select fast i1 %[[pinf]], double 0.000000e+00, double %differeturn
; CHECK-NEXT:   %[[dp1:.+]] = select fast i1 %[[adj]], double %[[dsel]], double 0.000000e+00
; CHECK-NEXT:   %[[drem1:.+]] = select fast i1 %[[adj]], double 0.000000e+00, double %[[dsel]]
; CHECK-NEXT:   %[[drem:.+]] = fadd fast double %[[drem1]], %[[dp1]]
; CHECK-NEXT:   %[[q:.+]] = fdiv fast double %a, %p
; CHECK-NEXT:   %[[qabs:.+]] = call fast double @llvm.fabs.f64(double %[[q]])
; CHECK-NEXT:   %[[qfl:.+]] = call fast double @llvm.floor.f64(double %[[qabs]])
; CHECK-NEXT:   %[[qtr:.+]] = call fast double @llvm.copysign.f64(double %[[qfl]], double %[[q]])
; CHECK-NEXT:   %[[nqtr:.+]] = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %[[qtr]]
; CHECK-NEXT:   %[[dp2:.+]] = fmul fast double %[[drem]], %[[nqtr]]
; CHECK-NEXT:   %[[dp:.+]] = fadd fast double %[[dp1]], %[[dp2]]
; CHECK-NEXT:   %[[r0:.+]] = insertvalue { double, double } undef, double %[[drem]], 0
; CHECK-NEXT:   %[[r1:.+]] = insertvalue { double, double } %[[r0]], double %[[dp]], 1
; CHECK-NEXT:   ret { double, double } %[[r1]]
