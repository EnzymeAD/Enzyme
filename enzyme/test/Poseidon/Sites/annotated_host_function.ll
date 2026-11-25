; REQUIRES: enzyme
; RUN: rm -rf %t && mkdir %t
; RUN: %opt %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,globaldce" -enzyme-preopt=false -poseidon-profile-generate -poseidon-cache=%t/gen -poseidon-print -S -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN-ERR %s
; RUN: FileCheck --check-prefix=GEN %s < %t/gen.ll
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -poseidon-cache=%t/use -S -o %t/use.ll 2>&1 | FileCheck --check-prefix=USE-ERR --implicit-check-not=warning: %s
; RUN: FileCheck --check-prefix=USE --implicit-check-not=_poseidon_body %s < %t/use.ll
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -S -o %t/off.ll 2>&1 | FileCheck --check-prefix=OFF-ERR --implicit-check-not=warning: %s
; RUN: FileCheck --check-prefix=OFF --implicit-check-not=_poseidon_body %s < %t/off.ll

; Each site's tables give the parameter number, the seed width and the name of
; each pointer argument; the globals table holds @table and its size.
; GEN: @table = internal global [4 x double] zeroinitializer
; GEN: [[XN:@[^ ]+]] = private unnamed_addr constant [2 x i8] c"x\00"
; GEN-NEXT: [[OUTN:@[^ ]+]] = private unnamed_addr constant [4 x i8] c"out\00"
; GEN-NEXT: [[SCN:@[^ ]+]] = private unnamed_addr constant [6 x i8] c"scale\00"
; GEN-NEXT: [[SCP:@[^ ]+]] = private unnamed_addr constant [2 x i32] [i32 1, i32 2]
; GEN-NEXT: [[SCW:@[^ ]+]] = private unnamed_addr constant [2 x i32] [i32 0, i32 8]
; GEN-NEXT: [[SCNAMES:@[^ ]+]] = private unnamed_addr constant [2 x ptr] [ptr [[XN]], ptr [[OUTN]]]
; GEN: [[DXN:@[^ ]+]] = private unnamed_addr constant [2 x i8] c"x\00"
; GEN-NEXT: [[DN:@[^ ]+]] = private unnamed_addr constant [4 x i8] c"dot\00"
; GEN-NEXT: [[DP:@[^ ]+]] = private unnamed_addr constant [1 x i32] [i32 1]
; GEN-NEXT: [[DW:@[^ ]+]] = private unnamed_addr constant [1 x i32] zeroinitializer
; GEN-NEXT: [[DNAMES:@[^ ]+]] = private unnamed_addr constant [1 x ptr] [ptr [[DXN]]]
; GEN-NEXT: [[TABLE:@[^ ]+]] = private unnamed_addr constant [1 x { ptr, i64 }] [{ ptr, i64 } { ptr @table, i64 32 }]
; GEN-NEXT: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @poseidon.register_static.preprocess_scale_poseidon_body, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon.register_static.preprocess_dot_poseidon_body, ptr null }, { i32, ptr, ptr } { i32 0, ptr [[GLOBALS:@[^ ]+]], ptr null }]

; GEN-LABEL: define void @scale(ptr %x, ptr %out)
; GEN-NEXT: entry:
; GEN-NEXT: [[POSEIDON_PTRS:%[^ ,()]+]] = alloca [2 x ptr]
; GEN-NEXT: [[POSEIDON_SHADOWS:%[^ ,()]+]] = alloca [2 x ptr]
; GEN-NEXT: [[P0:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[POSEIDON_PTRS]], {{i64 0|i32 0, i32 0}}
; GEN-NEXT: store ptr %x, ptr [[P0]]
; GEN-NEXT: [[P1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[POSEIDON_PTRS]], {{i64 8|i32 0, i32 1}}
; GEN-NEXT: store ptr %out, ptr [[P1]]
; GEN-NEXT: call void @__poseidon_host_shadows(ptr [[SCN]], i32 0, i32 2, ptr [[POSEIDON_PTRS]], ptr [[SCP]], ptr [[SCW]], ptr [[SCNAMES]], ptr [[POSEIDON_SHADOWS]])
; GEN-NEXT: [[S0:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[POSEIDON_SHADOWS]], {{i64 0|i32 0, i32 0}}
; GEN-NEXT: [[DX:%[0-9]+]] = load ptr, ptr [[S0]]
; GEN-NEXT: [[S1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[POSEIDON_SHADOWS]], {{i64 8|i32 0, i32 1}}
; GEN-NEXT: [[DOUT:%[0-9]+]] = load ptr, ptr [[S1]]
; GEN-NEXT: call void @diffepreprocess_scale_poseidon_body(ptr %x, ptr [[DX]], ptr %out, ptr [[DOUT]])
; GEN-NEXT: ret void
; GEN-LABEL: define double @dot(ptr %x, double %s, i32 %n)
; GEN-NEXT: entry:
; GEN-NEXT: [[POSEIDON_PTRS:%[^ ,()]+]] = alloca [1 x ptr]
; GEN-NEXT: [[POSEIDON_SHADOWS:%[^ ,()]+]] = alloca [1 x ptr]
; GEN-NEXT: [[P0:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[POSEIDON_PTRS]], {{i64 0|i32 0, i32 0}}
; GEN-NEXT: store ptr %x, ptr [[P0]]
; GEN-NEXT: call void @__poseidon_host_shadows(ptr [[DN]], i32 1, i32 1, ptr [[POSEIDON_PTRS]], ptr [[DP]], ptr [[DW]], ptr [[DNAMES]], ptr [[POSEIDON_SHADOWS]])
; GEN-NEXT: [[S0:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[POSEIDON_SHADOWS]], {{i64 0|i32 0, i32 0}}
; GEN-NEXT: [[DX:%[0-9]+]] = load ptr, ptr [[S0]]
; GEN-NEXT: [[R:%[0-9]+]] = call {{.*}}{ double, double } @diffepreprocess_dot_poseidon_body(ptr %x, ptr [[DX]], double %s, i32 %n, double 1.000000e+00)
; GEN-NEXT: [[P:%[0-9]+]] = extractvalue { double, double } [[R]], 0
; GEN-NEXT: ret double [[P]]
; GEN-LABEL: define double @caller(ptr %out)
; GEN-NEXT: %buf = alloca [4 x double]
; GEN-NEXT: call void @__poseidon_stack_push(ptr %buf, i64 32)
; GEN-NEXT: call void @scale(ptr %buf, ptr %out)
; GEN-NEXT: call void @__poseidon_stack_pop(i32 1)
; GEN-NEXT: call void @scale(ptr @table, ptr %out)
; GEN-NEXT: call void @__poseidon_stack_push(ptr %buf, i64 32)
; GEN-NEXT: call double @dot(ptr %buf, double 2.000000e+00, i32 4)
; GEN-NEXT: call void @__poseidon_stack_pop(i32 1)
; GEN-NEXT: ret double %d
; GEN: define internal void [[GLOBALS]]()
; GEN-NEXT: entry:
; GEN-NEXT: call void @__poseidon_register_globals(ptr [[TABLE]], i64 1)
; GEN-NEXT: ret void

; GEN-ERR: [poseidon] site dot: accuracy target 1e-07
; GEN-ERR: [poseidon] scale: site 0, 2 shadow buffer(s), output(s) 1, 1 call(s)
; GEN-ERR: [poseidon] dot: site 1, 1 shadow buffer(s), output(s) return, 1 call(s)

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
; USE-NEXT: [[V:%.+]] = load double, ptr %x
; USE-NEXT: [[M:%.+]] = fmul double [[V]], %s
; USE-NEXT: [[A:%.+]] = fadd double [[M]], %s
; USE-NEXT: ret double [[A]]
; USE-NOT: _poseidon_body
; USE-LABEL: define double @caller(ptr %out)
; USE-NEXT: %buf = alloca [4 x double]
; USE-NEXT: call void @scale(ptr %buf, ptr %out)
; USE-NEXT: call void @scale(ptr @table, ptr %out)
; USE-NEXT: %d = call double @dot(ptr %buf, double 2.000000e+00, i32 4)
; USE-NEXT: ret double %d
; USE-ERR: warning: <unknown>:0:0: in function scale void (ptr, ptr): Poseidon: no profile at {{.*}}/scale.fpprofile:
; USE-ERR: warning: <unknown>:0:0: in function dot double (ptr, double, i32): Poseidon: no profile at {{.*}}/dot.fpprofile:

; Without a profile action the module is the one written.
; OFF: define void @scale(ptr %x, ptr %out)
; OFF-NEXT: %v = load double, ptr %x
; OFF-NEXT: %m = fmul double %v, 3.000000e+00
; OFF-NEXT: %a = fadd double %m, 1.000000e+00
; OFF-NEXT: store double %a, ptr %out
; OFF-NEXT: ret void
; OFF: define double @dot(ptr %x, double %s, i32 %n)
; OFF-NEXT: %v = load double, ptr %x
; OFF-NEXT: %m = fmul double %v, %s
; OFF-NEXT: %a = fadd double %m, %s
; OFF-NEXT: ret double %a
; OFF: define double @caller(ptr %out)
; OFF-NEXT: %buf = alloca [4 x double]
; OFF-NEXT: call void @scale(ptr %buf, ptr %out)
; OFF-NEXT: call void @scale(ptr @table, ptr %out)
; OFF-NEXT: %d = call double @dot(ptr %buf, double 2.000000e+00, i32 4)
; OFF-NEXT: ret double %d
; OFF-NOT: _poseidon_body
; OFF-ERR: warning: <unknown>:0:0: in function scale void (ptr, ptr): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it
; OFF-ERR: warning: <unknown>:0:0: in function dot double (ptr, double, i32): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it

; POSEIDON_OPTIMIZE on a host function makes the function a wrapper around its
; outlined body, so every call of it is the site. Under profile generation
; each call of the body is a reverse-mode request whose shadows the runtime
; finds from the allocation each pointer argument points into; a stack object
; passed to a call is registered for that call, and the module's globals before
; main. A floating-point return is seeded by the request, a floating-point
; scalar parameter is active and an integer one constant.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"host.cpp\00", section "llvm.metadata"
@.args = private unnamed_addr constant { double } { double 1.000000e-07 }, section "llvm.metadata"
@table = internal global [4 x double] zeroinitializer
@llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @scale, ptr @.str, ptr @.str.1, i32 3, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @dot, ptr @.str, ptr @.str.1, i32 9, ptr @.args }], section "llvm.metadata"

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

define double @caller(ptr %out) {
  %buf = alloca [4 x double]
  call void @scale(ptr %buf, ptr %out)
  call void @scale(ptr @table, ptr %out)
  %d = call double @dot(ptr %buf, double 2.0, i32 4)
  ret double %d
}
