; DEFINE: %{clang} = %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -Wno-override-module %t/host.ll %loadClangPoseidon -S -emit-llvm
; DEFINE: %{use} = -poseidon-profile-use=%S/../Inputs/unrelated_site_profile
; RUN: rm -rf %t && split-file %s %t
; RUN: not %{clang} -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/gen -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: not %{clang} -mllvm %{use} -mllvm -poseidon-cache=%t/use -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: %opt %t/host.ll %loadPoseidon -passes=poseidon -S -o %t/off.ll 2>&1 | FileCheck --check-prefix=OFF --implicit-check-not=warning: %s
; RUN: FileCheck --check-prefix=OFFIR %s < %t/off.ll
; RUN: %opt %t/host.ll -S -o %t/in.ll
; RUN: sed -e 's/ #[0-9]* {$/ {/' -e '/^attributes #/d' %t/off.ll | diff -B %t/in.ll -
; RUN: not %opt %t/value.ll %loadPoseidon -passes=poseidon %{use} -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=VALUE %s
; RUN: not %opt %t/byref.ll %loadPoseidon -passes=poseidon %{use} -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=BYREF %s
; RUN: not %opt %t/callform.ll %loadPoseidon -passes="poseidon,poseidon-finalize" %{use} -poseidon-cache=%t/cf -S -o /dev/null 2>&1 | FileCheck --check-prefix=CALLFORM %s

; A function carrying POSEIDON_OPTIMIZE that cannot be a site. On the host
; (host.ll) every such function is an error under profile generation or use,
; and is compiled as written, marked as no site, without either; clang reports
; every error, where the opt of LLVM 21 and 22 stops at the first.
; ERR: error: <unknown>:0:0: in function pair { double, double } (double): Poseidon: pair returns a value that is not a floating-point or integer scalar, so it cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function vec <2 x double> (<2 x double>): Poseidon: vec returns a value that is not a floating-point or integer scalar, so it cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function viasret void (ptr, double): Poseidon: viasret returns its value through memory (sret), so it cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function variadic double (double, ...): Poseidon: variadic is variadic, so it cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function recur double (double): Poseidon: recur calls itself, so it cannot be a Poseidon site
; ERR: 5 errors generated.
; OFF: warning: <unknown>:0:0: in function pair { double, double } (double): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it
; OFF: warning: <unknown>:0:0: in function vec <2 x double> (<2 x double>): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it
; OFF: warning: <unknown>:0:0: in function viasret void (ptr, double): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it
; OFF: warning: <unknown>:0:0: in function variadic double (double, ...): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it
; OFF: warning: <unknown>:0:0: in function recur double (double): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it
; OFF-NOT: error
; OFFIR: define { double, double } @pair(double %x) [[NAS:#[0-9]+]] {
; OFFIR: define <2 x double> @vec(<2 x double> %x) [[NAS]] {
; OFFIR: define void @viasret(ptr sret({ double, double }) %r, double %x) [[NAS]] {
; OFFIR: define double @variadic(double %x, ...) [[NAS]] {
; OFFIR: define double @recur(double %x) [[NAS]] {
; OFFIR: attributes [[NAS]] = { "poseidon-not-a-site" }

; A kernel that returns a value, and an AMDGPU kernel that uses an aggregate
; it takes by reference otherwise than by loading or copying from it.
; VALUE: error: {{.*}}Poseidon: an annotated kernel must be a void, non-variadic function
; BYREF: error: {{.*}}Poseidon: k uses its aggregate parameter 1 otherwise than by loading or copying from it, so it cannot be a Poseidon site; copy the parameter into a local object first

; Naming an annotated function in the call form as well is an error at the call.
; CALLFORM: error: {{.*}}in function site {{.*}}Poseidon: tester carries POSEIDON_OPTIMIZE, so every call of it is already a site; remove this __poseidon_optimize or the annotation

;--- host.ll
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

;--- value.ll
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [5 x i8] c"k.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @k, ptr @.str, ptr @.str.1, i32 1, ptr null }], section "llvm.metadata"

define ptx_kernel double @k(ptr %x) {
  %v = load double, ptr %x
  %m = fmul double %v, %v
  ret double %m
}

;--- byref.ll
target triple = "amdgcn-amd-amdhsa"

%pair = type { ptr, double }

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [6 x i8] c"k.hip\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @k, ptr @.str, ptr @.str.1, i32 1, ptr null }], section "llvm.metadata"

define amdgpu_kernel void @k(ptr addrspace(4) byref(%pair) align 8 %p) {
  %x = load ptr, ptr addrspace(4) %p
  %s = getelementptr inbounds %pair, ptr addrspace(4) %p, i32 0, i32 1
  %v = load double, ptr addrspace(4) %s
  %m = fmul double %v, %v
  store double %m, ptr %x
  call void @keep(ptr addrspace(4) %p)
  ret void
}

declare void @keep(ptr addrspace(4))

;--- callform.ll
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"host.cpp\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @tester, ptr @.str, ptr @.str.1, i32 3, ptr null }], section "llvm.metadata"

define double @tester(double %x, double %y) {
  %m = fmul double %x, %y
  %a = fadd double %m, %x
  ret double %a
}

define double @site(double %x, double %y) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @tester, double %x, double %y)
  ret double %r
}

declare double @__poseidon_fp_optimize(ptr, ...)
