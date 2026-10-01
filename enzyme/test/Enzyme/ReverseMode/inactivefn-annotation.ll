; The clang plugin lowers __attribute__((enzyme_inactive)) on a templated
; function to an enzyme_inactivefn annotation rather than a registration global,
; since a templated declaration has no address until it is instantiated. Unlike
; a bare enzyme_inactive annotation, that also marks the body of the function.

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -preserve-nvvm -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="preserve-nvvm" -S | FileCheck %s

source_filename = "inactivefn-annotation.ll"

@.str.blocking = private unnamed_addr constant [18 x i8] c"enzyme_inactivefn\00", section "llvm.metadata"
@.str.noblock = private unnamed_addr constant [25 x i8] c"enzyme_inactivenoblockfn\00", section "llvm.metadata"
@.str.plain = private unnamed_addr constant [16 x i8] c"enzyme_inactive\00", section "llvm.metadata"
@.str.file = private unnamed_addr constant [25 x i8] c"inactivefn-annotation.ll\00", section "llvm.metadata"

@llvm.global.annotations = appending global [3 x { i8*, i8*, i8*, i32, i8* }] [
  { i8*, i8*, i8*, i32, i8* } { i8* bitcast (double (double)* @blocking to i8*), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.blocking, i32 0, i32 0), i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.file, i32 0, i32 0), i32 1, i8* null },
  { i8*, i8*, i8*, i32, i8* } { i8* bitcast (double (double)* @noblock to i8*), i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.noblock, i32 0, i32 0), i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.file, i32 0, i32 0), i32 2, i8* null },
  { i8*, i8*, i8*, i32, i8* } { i8* bitcast (double (double)* @plain to i8*), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.plain, i32 0, i32 0), i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.file, i32 0, i32 0), i32 3, i8* null }
], section "llvm.metadata"

define double @blocking(double %a) {
entry:
  %mul = fmul double %a, 2.000000e+00
  ret double %mul
}

define double @noblock(double %a) {
entry:
  %mul = fmul double %a, 3.000000e+00
  ret double %mul
}

define double @plain(double %a) {
entry:
  %mul = fmul double %a, 4.000000e+00
  ret double %mul
}

; The annotations are consumed, so the array is left zeroed.
; CHECK: @llvm.global.annotations = appending global [3 x { i8*, i8*, i8*, i32, i8* }] zeroinitializer

; enzyme_inactivefn marks the function and its body.
; CHECK-LABEL: define double @blocking(double %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %a, 2.000000e+00, !enzyme_inactive
; CHECK-NEXT:   ret double %mul, !enzyme_inactive

; enzyme_inactivenoblockfn currently lowers the same way.
; CHECK-LABEL: define double @noblock(double %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %a, 3.000000e+00, !enzyme_inactive
; CHECK-NEXT:   ret double %mul, !enzyme_inactive

; A bare enzyme_inactive annotation marks only the function itself.
; CHECK-LABEL: define double @plain(double %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %a, 4.000000e+00
; CHECK-NEXT:   ret double %mul

; All three are inactive at the function level.
; CHECK: attributes #{{[0-9]+}} = {{.*}}"enzyme_inactive"
