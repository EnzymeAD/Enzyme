; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,globaldce" -enzyme-preopt=false -poseidon-profile-generate -poseidon-cache=%t/gen -S -o %t/gen.ll 2> %t/gen.err
; RUN: FileCheck --check-prefix=GEN %s < %t/gen.ll
; RUN: FileCheck --check-prefix=GEN-ERR %s < %t/gen.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" -poseidon-profile-use=%t/none -poseidon-cache=%t/use -S -o %t/use.ll 2> %t/use.err
; RUN: FileCheck --check-prefix=USE %s < %t/use.ll
; RUN: FileCheck --check-prefix=USE-ERR %s < %t/use.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -S -o %t/off.ll 2> %t/off.err
; RUN: FileCheck --check-prefix=OFF %s < %t/off.ll
; RUN: FileCheck --check-prefix=OFF-ERR %s < %t/off.err
; REQUIRES: poseidon, enzyme

; POSEIDON_OPTIMIZE on a host function: the function becomes a wrapper around
; its outlined body, so every call of it is the site. Under profile generation
; each call of the body is a reverse-mode request whose shadows the runtime
; finds from the allocation each pointer argument points into; a stack object
; passed to a call is registered for that call, and the module's globals before
; main. A floating-point return is seeded by the request, a floating-point
; scalar parameter is active and an integer one constant. A function whose
; signature has no such activity is not a site.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"host.cpp\00", section "llvm.metadata"
@.args = private unnamed_addr constant { double } { double 1.000000e-07 }, section "llvm.metadata"
@table = internal global [4 x double] zeroinitializer
@llvm.global.annotations = appending global [7 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @scale, ptr @.str, ptr @.str.1, i32 3, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @dot, ptr @.str, ptr @.str.1, i32 9, ptr @.args },
  { ptr, ptr, ptr, i32, ptr } { ptr @pair, ptr @.str, ptr @.str.1, i32 15, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @vec, ptr @.str, ptr @.str.1, i32 20, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @viasret, ptr @.str, ptr @.str.1, i32 24, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @variadic, ptr @.str, ptr @.str.1, i32 28, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @recur, ptr @.str, ptr @.str.1, i32 32, ptr null }], section "llvm.metadata"

define void @scale(ptr %x, ptr %out) {
  %v = load double, ptr %x
  %m = fmul double %v, 3.000000e+00
  %a = fadd double %m, 1.000000e+00
  store double %a, ptr %out
  ret void
}

define double @dot(ptr %x, double %s, i32 %n) {
  %v = load double, ptr %x
  %m = fmul double %v, %s
  %a = fadd double %m, %s
  ret double %a
}

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

define double @caller(ptr %out) {
  %buf = alloca [4 x double]
  call void @scale(ptr %buf, ptr %out)
  call void @scale(ptr @table, ptr %out)
  %d = call double @dot(ptr %buf, double 2.0, i32 4)
  ret double %d
}

; GEN-LABEL: define void @scale(ptr %x, ptr %out)
; GEN: call void @__poseidon_host_shadows(ptr @{{[^,]+}}, i32 0, i32 2, ptr %poseidon.ptrs, {{.*}}, ptr %poseidon.shadows)
; GEN: call void @diffepreprocess_scale_poseidon_body(ptr %x, ptr %{{[0-9]+}}, ptr %out, ptr %{{[0-9]+}})
; GEN-LABEL: define double @dot(ptr %x, double %s, i32 %n)
; GEN: call void @__poseidon_host_shadows(ptr @{{[^,]+}}, i32 1, i32 1,
; GEN: [[R:%[0-9]+]] = call {{.*}}{ double, double } @diffepreprocess_dot_poseidon_body(ptr %x, ptr %{{[0-9]+}}, double %s, i32 %n, double 1.000000e+00)
; GEN: [[P:%[0-9]+]] = extractvalue { double, double } [[R]], 0
; GEN: ret double [[P]]
; GEN-LABEL: define { double, double } @pair(double %x)
; GEN-NEXT: %m = fmul double %x, %x
; GEN-LABEL: define double @caller(ptr %out)
; GEN: call void @__poseidon_stack_push(ptr %buf, i64 32)
; GEN-NEXT: call void @scale(ptr %buf, ptr %out)
; GEN-NEXT: call void @__poseidon_stack_pop(i32 1)
; GEN-NEXT: call void @scale(ptr @table, ptr %out)
; GEN-NEXT: call void @__poseidon_stack_push(ptr %buf, i64 32)
; GEN-NEXT: call double @dot(ptr %buf, double 2.000000e+00, i32 4)
; GEN-NEXT: call void @__poseidon_stack_pop(i32 1)
; GEN: call void @__poseidon_register_globals(ptr [[TABLE:@[^,]+]], i64 1)

; GEN-ERR: warning: {{.*}}Poseidon: pair returns a value that is not a floating-point or integer scalar, so it is not a Poseidon site
; GEN-ERR: warning: {{.*}}Poseidon: vec returns a value that is not a floating-point or integer scalar, so it is not a Poseidon site
; GEN-ERR: warning: {{.*}}Poseidon: viasret returns its value through memory (sret), so it is not a Poseidon site
; GEN-ERR: warning: {{.*}}Poseidon: variadic is variadic, so it is not a Poseidon site
; GEN-ERR: warning: {{.*}}Poseidon: recur calls itself, so it is not a Poseidon site
; GEN-ERR: [poseidon-profgen] scale: site 0, 2 shadow buffer(s), output(s) 1, 1 call(s)
; GEN-ERR: [poseidon-profgen] dot: site 1, 1 shadow buffer(s), output(s) return, 1 call(s)

; Without a profile the site keeps the body it was written with, folded back.
; USE-LABEL: define void @scale(ptr %x, ptr %out)
; USE-NEXT: entry:
; USE-NEXT: [[V:%.+]] = load double, ptr %x
; USE-NEXT: [[M:%.+]] = fmul double [[V]], 3.000000e+00
; USE-NEXT: [[A:%.+]] = fadd double [[M]], 1.000000e+00
; USE-NEXT: store double [[A]], ptr %out
; USE-NEXT: ret void
; USE-LABEL: define double @dot(ptr %x, double %s, i32 %n)
; USE-NEXT: entry:
; USE-NEXT: load double, ptr %x
; USE-NOT: _poseidon_body
; USE-ERR: scale_poseidon_body: no profile at {{.*}}; left unchanged
; USE-ERR: dot_poseidon_body: no profile at {{.*}}; left unchanged

; OFF: define void @scale(ptr %x, ptr %out)
; OFF-NEXT: %v = load double, ptr %x
; OFF-NOT: _poseidon_body
; OFF-ERR: warning: {{.*}}in function scale {{.*}}compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>
; OFF-ERR: warning: {{.*}}in function dot {{.*}}compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>
