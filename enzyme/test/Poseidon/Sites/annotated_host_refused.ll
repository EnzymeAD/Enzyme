; RUN: rm -rf %t && mkdir %t
; RUN: not %clang -O0 -x ir -Wno-override-module %s %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/gen -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: not %clang -O0 -x ir -Wno-override-module %s %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/use -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: %opt %s %loadPoseidon -passes=poseidon -S -o %t/off.ll 2>&1 | FileCheck --check-prefix=OFF --implicit-check-not=warning: %s
; RUN: FileCheck --check-prefix=OFFIR %s < %t/off.ll

; A host function carrying POSEIDON_OPTIMIZE whose signature has no activity
; for every parameter and its value (annotated_host_function.ll) cannot be a
; site: in a compile that generates or uses a profile that is an error, every
; such function reported. Without either the function compiles as written.
; The opt of LLVM 21 and 22 exits at its first error, so the refusing compiles
; run through clang, which reports every error of a compile.

; ERR: error: <unknown>:0:0: in function pair { double, double } (double): Poseidon: pair returns a value that is not a floating-point or integer scalar, so it cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function vec <2 x double> (<2 x double>): Poseidon: vec returns a value that is not a floating-point or integer scalar, so it cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function viasret void (ptr, double): Poseidon: viasret returns its value through memory (sret), so it cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function variadic double (double, ...): Poseidon: variadic is variadic, so it cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function recur double (double): Poseidon: recur calls itself, so it cannot be a Poseidon site
; ERR: 5 errors generated.
; OFF: warning: <unknown>:0:0: in function pair { double, double } (double): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>. Emitting the original computation.
; OFF: warning: <unknown>:0:0: in function vec <2 x double> (<2 x double>): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>. Emitting the original computation.
; OFF: warning: <unknown>:0:0: in function viasret void (ptr, double): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>. Emitting the original computation.
; OFF: warning: <unknown>:0:0: in function variadic double (double, ...): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>. Emitting the original computation.
; OFF: warning: <unknown>:0:0: in function recur double (double): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>. Emitting the original computation.
; OFF-NOT: error

; Each body stays as written; the pass only marks the function as no site.
; OFFIR: define { double, double } @pair(double %x) [[NAS:#[0-9]+]] {
; OFFIR-NEXT: %m = fmul double %x, %x
; OFFIR-NEXT: %r = insertvalue { double, double } undef, double %m, 0
; OFFIR-NEXT: ret { double, double } %r
; OFFIR: define <2 x double> @vec(<2 x double> %x) [[NAS]] {
; OFFIR-NEXT: %m = fmul <2 x double> %x, %x
; OFFIR-NEXT: ret <2 x double> %m
; OFFIR: define void @viasret(ptr sret({ double, double }) %r, double %x) [[NAS]] {
; OFFIR-NEXT: %m = fmul double %x, %x
; OFFIR-NEXT: store double %m, ptr %r
; OFFIR-NEXT: ret void
; OFFIR: define double @variadic(double %x, ...) [[NAS]] {
; OFFIR-NEXT: %m = fmul double %x, %x
; OFFIR-NEXT: ret double %m
; OFFIR: define double @recur(double %x) [[NAS]] {
; OFFIR-NEXT: %m = fmul double %x, %x
; OFFIR-NEXT: %c = fcmp olt double %m, 1.000000e+00
; OFFIR-NEXT: br i1 %c, label %done, label %again
; OFFIR-EMPTY:
; OFFIR-NEXT: again:
; OFFIR-NEXT: %h = fmul double %x, 5.000000e-01
; OFFIR-NEXT: %r = call double @recur(double %h)
; OFFIR-NEXT: ret double %r
; OFFIR-EMPTY:
; OFFIR-NEXT: done:
; OFFIR-NEXT: ret double %m
; OFFIR: attributes [[NAS]] = { "poseidon-not-a-site" }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"host.cpp\00", section "llvm.metadata"
@llvm.global.annotations = appending global [5 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @pair, ptr @.str, ptr @.str.1, i32 15, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @vec, ptr @.str, ptr @.str.1, i32 20, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @viasret, ptr @.str, ptr @.str.1, i32 24, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @variadic, ptr @.str, ptr @.str.1, i32 28, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @recur, ptr @.str, ptr @.str.1, i32 32, ptr null }], section "llvm.metadata"

define { double, double } @pair(double %x) {
  %m = fmul double %x, %x
  %r = insertvalue { double, double } undef, double %m, 0
  ret { double, double } %r
}

define <2 x double> @vec(<2 x double> %x) {
  %m = fmul <2 x double> %x, %x
  ret <2 x double> %m
}

define void @viasret(ptr sret({ double, double }) %r, double %x) {
  %m = fmul double %x, %x
  store double %m, ptr %r
  ret void
}

define double @variadic(double %x, ...) {
  %m = fmul double %x, %x
  ret double %m
}

define double @recur(double %x) {
  %m = fmul double %x, %x
  %c = fcmp olt double %m, 1.000000e+00
  br i1 %c, label %done, label %again
again:
  %h = fmul double %x, 5.000000e-01
  %r = call double @recur(double %h)
  ret double %r
done:
  ret double %m
}
