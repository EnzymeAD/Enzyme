; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-generate -S 2>%t.gen.err > %t.gen.ll
; RUN: FileCheck %s < %t.gen.ll
; RUN: FileCheck --check-prefix=NOAD %s < %t.gen.ll
; RUN: FileCheck --check-prefix=WARN %s < %t.gen.err
; RUN: rm -rf %t && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/profile -poseidon-cache=%t -S 2>%t.use.err | FileCheck %s
; RUN: FileCheck --check-prefix=WARN %s < %t.use.err
; REQUIRES: poseidon
; POSEIDON_OPTIMIZE on a host function: a site's profiled clone takes shadow
; arguments its callers never pass, so the function is not a site. It and its
; call stay as written, with one warning per compile.

target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"host.cpp\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @scale, ptr @.str, ptr @.str.1, i32 3, ptr null }], section "llvm.metadata"

define void @scale(ptr %x, ptr %out) {
  %v = load double, ptr %x
  %m = fmul double %v, 3.000000e+00
  store double %m, ptr %out
  ret void
}

define void @caller(ptr %x, ptr %out) {
  call void @scale(ptr %x, ptr %out)
  ret void
}

; CHECK: define void @scale(ptr %x, ptr %out)
; CHECK-NEXT: %v = load double, ptr %x
; CHECK-NEXT: %m = fmul double %v, 3.000000e+00
; CHECK: define void @caller(ptr %x, ptr %out)
; CHECK-NEXT: call void @scale(ptr %x, ptr %out)

; NOAD-NOT: __enzyme_autodiff_poseidon

; WARN: warning: {{.*}}POSEIDON_OPTIMIZE marks GPU kernels; on the host, wrap the call in __poseidon_fp_optimize
; WARN-NOT: POSEIDON_OPTIMIZE marks GPU kernels
