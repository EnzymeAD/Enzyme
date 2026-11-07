; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm" -S | FileCheck %s; fi

; preserve-nvvm lowers flang's runtime calls for MOD and MODULO of REAL(4),
; REAL(8) and REAL(10) to frem. REAL(16) (fp128) keeps the runtime call:
; frem on fp128 may need a libcall the target does not have. A declaration
; whose signature differs from the runtime's is left alone.

@file = private constant [6 x i8] c"m.f90\00"

define float @mod4(float %a, float %p) {
entry:
  %r = call float @_FortranAModReal4(float %a, float %p, ptr @file, i32 1)
  ret float %r
}

define x86_fp80 @modulo10(x86_fp80 %a, x86_fp80 %p) {
entry:
  %r = call x86_fp80 @_FortranAModuloReal10(x86_fp80 %a, x86_fp80 %p, ptr @file, i32 2)
  ret x86_fp80 %r
}

define fp128 @mod16(fp128 %a, fp128 %p) {
entry:
  %r = call fp128 @_FortranAModReal16(fp128 %a, fp128 %p, ptr @file, i32 3)
  ret fp128 %r
}

define double @modulo8_badsig(double %a, double %p) {
entry:
  %r = call double @_FortranAModuloReal8(double %a, double %p)
  ret double %r
}

declare float @_FortranAModReal4(float, float, ptr, i32)
declare x86_fp80 @_FortranAModuloReal10(x86_fp80, x86_fp80, ptr, i32)
declare fp128 @_FortranAModReal16(fp128, fp128, ptr, i32)
declare double @_FortranAModuloReal8(double, double)

; CHECK: define float @mod4(float %a, float %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r = frem float %a, %p
; CHECK-NEXT:   ret float %r

; CHECK: define x86_fp80 @modulo10(x86_fp80 %a, x86_fp80 %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[rem:.+]] = frem x86_fp80 %a, %p
; CHECK-NEXT:   %[[pneg:.+]] = fcmp olt x86_fp80 %p, 0xK00000000000000000000
; CHECK-NEXT:   %[[aneg:.+]] = fcmp olt x86_fp80 %a, 0xK00000000000000000000
; CHECK-NEXT:   %[[differ:.+]] = xor i1 %[[aneg]], %[[pneg]]
; CHECK-NEXT:   %[[nz:.+]] = fcmp une x86_fp80 %[[rem]], 0xK00000000000000000000
; CHECK-NEXT:   %[[adj:.+]] = and i1 %[[nz]], %[[differ]]
; CHECK-NEXT:   %[[radd:.+]] = fadd x86_fp80 %[[rem]], %p
; CHECK-NEXT:   %[[sel:.+]] = select i1 %[[adj]], x86_fp80 %[[radd]], x86_fp80 %[[rem]]
; CHECK-NEXT:   %[[pabs:.+]] = call x86_fp80 @llvm.fabs.f80(x86_fp80 %p)
; CHECK-NEXT:   %[[pinf:.+]] = fcmp oeq x86_fp80 %[[pabs]], 0xK7FFF8000000000000000
; CHECK-NEXT:   %r = select i1 %[[pinf]], x86_fp80 0xK7FFFC000000000000000, x86_fp80 %[[sel]]
; CHECK-NEXT:   ret x86_fp80 %r

; CHECK: define fp128 @mod16(fp128 %a, fp128 %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r = call fp128 @_FortranAModReal16(fp128 %a, fp128 %p, ptr @file, i32 3)

; CHECK: define double @modulo8_badsig(double %a, double %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r = call double @_FortranAModuloReal8(double %a, double %p)
