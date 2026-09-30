; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

; A gc_preserve_begin may legally end up with several gc_preserve_ends: the
; token it produces may have many uses, so LLVM is free to clone a block holding
; one of the ends. Reverse mode mirrors the region by emitting a begin at the
; invert of the end and the end at the invert of the begin, which only
; type-checks if a single end post-dominates the begin, so the ends have to be
; merged before differentiating. Every function below aborts in the verifier if
; they are not.

declare token @llvm.julia.gc_preserve_begin(...)

declare void @llvm.julia.gc_preserve_end(token)

declare void @throw() noreturn

; The two ends rejoin at a block the begin dominates.
define void @f({} addrspace(10)* %z, i1 %c) {
entry:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %a, label %b

a:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

join:
  ret void
}

; The two ends only rejoin at a block that is also reachable without ever
; running the begin, so the edges coming out of the region have to be split off
; into a block of their own first.
define void @g({} addrspace(10)* %z, i1 %c, i1 %d) {
entry:
  br i1 %d, label %pre, label %join

pre:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %a, label %b

a:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

join:
  ret void
}

; One end sits in the block the other rejoins, so the merged end has to take
; the place of that end rather than go at the top of its block: the store in
; between was inside the region on one path and must stay there.
define void @h({} addrspace(10)* %z, i1 %c) {
entry:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  br i1 %c, label %a, label %b

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %a

a:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret void
}

; An end on a path that only ever aborts is not visited by the reverse pass and
; must neither take part in the merge nor prevent it.
define void @k({} addrspace(10)* %z, i1 %c, i1 %e) {
entry:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %a, label %b

a:
  br i1 %e, label %a2, label %err

a2:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

err:
  call void @llvm.julia.gc_preserve_end(token %tok)
  call void @throw()
  unreachable

join:
  ret void
}

; A block that only aborts, shared between the region and code outside of it,
; is not a way out of the region that the merged end could miss.
define void @m({} addrspace(10)* %z, i1 %c, i1 %d, i1 %e) {
entry:
  br i1 %d, label %pre, label %check

check:
  br i1 %e, label %join, label %err

pre:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %a, label %b

a:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br i1 %e, label %join, label %err

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

err:
  call void @throw()
  unreachable

join:
  ret void
}

; The whole region lives inside a loop body.
define void @loop({} addrspace(10)* %z, i1 %c, i64 %n) {
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %a, label %b

a:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %latch

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %latch

latch:
  %inc = add i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %header

exit:
  ret void
}

; Each end was cloned together with the return that followed it, so nothing
; short of the function exit post-dominates them: the returns have to be
; unified first, and the merged end then goes into the unified return block.
define double @multiret({} addrspace(10)* %z, i1 %c) {
entry:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %a, label %b

a:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret double 1.000000e+00

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  ret double 2.000000e+00
}

; Only one end is on a returning path, so nothing needs merging, and the
; returns and throw blocks must be left as they are.
define double @throwret({} addrspace(10)* %z, i1 %c, i1 %d) {
entry:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  %v = load double, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %err, label %chk

err:
  call void @llvm.julia.gc_preserve_end(token %tok)
  call void @throw()
  unreachable

chk:
  br i1 %d, label %err2, label %ok

err2:
  call void @throw()
  unreachable

ok:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br i1 %d, label %r1, label %r2

r1:
  ret double %v

r2:
  %w = fmul double %v, %v
  ret double %w
}

; Function Attrs: nounwind
declare i8* @__enzyme_virtualreverse(...)

define i8* @test({} addrspace(10)* %z, i1 %c, i1 %d, i1 %e, i64 %n) {
entry:
  %0 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*, i1)* @f)
  %1 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*, i1, i1)* @g)
  %2 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*, i1)* @h)
  %3 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*, i1, i1)* @k)
  %4 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*, i1, i1, i1)* @m)
  %5 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*, i1, i64)* @loop)
  %6 = call i8* (...) @__enzyme_virtualreverse(double ({} addrspace(10)*, i1)* @multiret)
  %7 = call i8* (...) @__enzyme_virtualreverse(double ({} addrspace(10)*, i1, i1)* @throwret)
  ret i8* %0
}

; CHECK-LABEL: define internal {{(i8\*|ptr)}} @augmented_f(
; CHECK: %[[F_TOK:.+]] = call token (...) @llvm.julia.gc_preserve_begin(
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: join:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[F_TOK]])
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: ret

; CHECK-LABEL: define internal void @diffef(

; CHECK-LABEL: define internal {{(i8\*|ptr)}} @augmented_g(
; CHECK: %[[G_TOK:.+]] = call token (...) @llvm.julia.gc_preserve_begin(
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: join.gcpreserve:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[G_TOK]])
; CHECK-NEXT: br label %join
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: ret

; CHECK-LABEL: define internal void @diffeg(

; CHECK-LABEL: define internal {{(i8\*|ptr)}} @augmented_h(
; CHECK: %[[H_TOK:.+]] = call token (...) @llvm.julia.gc_preserve_begin(
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: a:
; CHECK-NEXT: store double 2.710000e+00
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[H_TOK]])
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: ret

; CHECK-LABEL: define internal void @diffeh(

; CHECK-LABEL: define internal {{(i8\*|ptr)}} @augmented_k(
; CHECK: %[[K_TOK:.+]] = call token (...) @llvm.julia.gc_preserve_begin(
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: err:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[K_TOK]])
; CHECK-NEXT: call void @throw()
; CHECK-NEXT: unreachable
; CHECK: join:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[K_TOK]])
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: ret

; CHECK-LABEL: define internal void @diffek(

; CHECK-LABEL: define internal {{(i8\*|ptr)}} @augmented_m(
; CHECK: %[[M_TOK:.+]] = call token (...) @llvm.julia.gc_preserve_begin(
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: join.gcpreserve:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[M_TOK]])
; CHECK-NEXT: br label %join
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: ret

; CHECK-LABEL: define internal void @diffem(

; CHECK-LABEL: define internal {{(i8\*|ptr)}} @augmented_loop(
; CHECK: %[[L_TOK:.+]] = call token (...) @llvm.julia.gc_preserve_begin(
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: latch:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[L_TOK]])
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: ret

; CHECK-LABEL: define internal void @diffeloop(

; CHECK-LABEL: define internal { {{(i8\*|ptr)}}, double } @augmented_multiret(
; CHECK: %[[R_TOK:.+]] = call token (...) @llvm.julia.gc_preserve_begin(
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: UnifiedReturnBlock:
; CHECK-NEXT: %[[R_RET:.+]] = phi double [ 1.000000e+00, %a ], [ 2.000000e+00, %b ]
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[R_TOK]])
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: ret

; CHECK-LABEL: define internal void @diffemultiret(

; CHECK-LABEL: define internal { {{(i8\*|ptr)}}, double } @augmented_throwret(
; CHECK-NOT: Unified
; CHECK: err:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(
; CHECK-NEXT: call void @throw()
; CHECK-NEXT: unreachable
; CHECK: err2:
; CHECK-NEXT: call void @throw()
; CHECK-NEXT: unreachable
; CHECK: ok:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(
; CHECK-NOT: Unified
; CHECK: ret { {{(i8\*|ptr)}}, double }
; CHECK-NOT: Unified
; CHECK: ret { {{(i8\*|ptr)}}, double }

; CHECK-LABEL: define internal void @diffethrowret(
