; REQUIRES: enzyme_test_plugin
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme %loadTestCallHandler -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme %loadTestCallHandler -enzyme-preopt=false -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme %loadTestCallHandler -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

;;; Verify that `GradientUtils::cacheForReverse` rebuilds primal and shadow
;;; function arguments directly in the split reverse pass rather than allocating
;;; physical tape slots or crashing when no defining instruction exists.

@.str = private unnamed_addr constant [6 x i8] c"%.1f\0A\00", align 1
@__enzyme_test_sink = internal global i32 0, align 4

declare void @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)

define ptr @__enzyme_test_arg_shadow(ptr %p) noinline {
entry:
  store i32 0, ptr @__enzyme_test_sink, align 4
  ret ptr %p
}

define void @__enzyme_test_arg_tape(ptr %out, double %scale) noinline {
entry:
  %val = load double, ptr %out, align 8
  %scaled = fmul double %val, %scale
  store double %scaled, ptr %out, align 8
  ret void
}

define double @__enzyme_test_arg_self(double %s) noinline {
entry:
  store i32 0, ptr @__enzyme_test_sink, align 4
  ret double %s
}

;;; 1. Only `cacheForReverse` value is a shadow argument (`CacheType::Shadow`):
;;;    no physical tape should be created (`void @augmented_...`, no tape arg on
;;;    `@diffe...`).
define void @callee_shadow_only(ptr %x, ptr %out) noinline {
entry:
  %slot = call ptr @__enzyme_test_arg_shadow(ptr %x)
  %vx = load double, ptr %slot, align 8
  %twice = fmul double %vx, 2.000000e+00
  store double %twice, ptr %out, align 8
  ret void
}

define void @top_shadow_only(ptr %x, ptr %out) {
entry:
  call void @callee_shadow_only(ptr %x, ptr %out)
  %y = load double, ptr %out, align 8
  %res = fmul double %y, %y
  store double %res, ptr %out, align 8
  ret void
}

;;; 2. Shadow argument alongside a single overwritten load: `removeTapeStruct`
;;;    must collapse the 1-element tape to scalar `double` while preserving the
;;;    negative uncached-argument entry in `tapeIndices`.
define void @callee_shadow_scalar_tape(ptr %x, ptr %buf, ptr %out) noinline {
entry:
  %slot = call ptr @__enzyme_test_arg_shadow(ptr %x)
  %vx = load double, ptr %slot, align 8
  %w = load double, ptr %buf, align 8
  store double 0.000000e+00, ptr %buf, align 8
  %prod = fmul double %vx, %w
  store double %prod, ptr %out, align 8
  ret void
}

define void @top_shadow_scalar_tape(ptr %x, ptr %buf, ptr %out) {
entry:
  call void @callee_shadow_scalar_tape(ptr %x, ptr %buf, ptr %out)
  %y = load double, ptr %out, align 8
  %res = fmul double %y, 3.000000e+00
  store double %res, ptr %out, align 8
  ret void
}

;;; 3. Shadow argument alongside two overwritten loads: the tape struct must
;;;    contain only the two physical `double` slots (`{ double, double }`).
define void @callee_shadow_struct_tape(ptr %x, ptr %buf1, ptr %buf2, ptr %out) noinline {
entry:
  %slot = call ptr @__enzyme_test_arg_shadow(ptr %x)
  %vx = load double, ptr %slot, align 8
  %w1 = load double, ptr %buf1, align 8
  store double 0.000000e+00, ptr %buf1, align 8
  %w2 = load double, ptr %buf2, align 8
  store double 0.000000e+00, ptr %buf2, align 8
  %p1 = fmul double %vx, %w1
  %p2 = fmul double %p1, %w2
  store double %p2, ptr %out, align 8
  ret void
}

define void @top_shadow_struct_tape(ptr %x, ptr %buf1, ptr %buf2, ptr %out) {
entry:
  call void @callee_shadow_struct_tape(ptr %x, ptr %buf1, ptr %buf2, ptr %out)
  %y = load double, ptr %out, align 8
  %res = fmul double %y, 2.000000e+00
  store double %res, ptr %out, align 8
  ret void
}

;;; 4. Shadow argument inside a loop: rebuilding the invariant shadow argument
;;;    must not allocate a loop tape in the augmented forward pass.
define void @callee_shadow_in_loop(ptr %x, ptr %out) noinline {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %slot = call ptr @__enzyme_test_arg_shadow(ptr %x)
  %vx = load double, ptr %slot, align 8
  %acc = load double, ptr %out, align 8
  %sum = fadd double %acc, %vx
  store double %sum, ptr %out, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp eq i64 %iv.next, 4
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

define void @top_shadow_in_loop(ptr %x, ptr %out) {
entry:
  call void @callee_shadow_in_loop(ptr %x, ptr %out)
  %y = load double, ptr %out, align 8
  %sq = fmul double %y, %y
  store double %sq, ptr %out, align 8
  ret void
}

;;; 5. Primal argument passed through custom `*tape` (`CacheType::Tape`): no
;;;    physical tape should be emitted and the split reverse pass must rebuild
;;;    the primal scalar argument directly.
define void @callee_primal_tape(ptr %x, double %scale) noinline {
entry:
  call void @__enzyme_test_arg_tape(ptr %x, double %scale)
  ret void
}

define void @top_primal_tape(ptr %x, double %scale) {
entry:
  call void @callee_primal_tape(ptr %x, double %scale)
  %y = load double, ptr %x, align 8
  %sq = fmul double %y, %y
  store double %sq, ptr %x, align 8
  ret void
}

;;; 6. Primal argument replacing `*normalReturn` (`CacheType::Self` with
;;;    `replace = true`): erasing the call in the split gradient must preserve
;;;    `originalToNewFn` so downstream reverse instructions find the rebuilt
;;;    argument without allocating a tape.
define void @callee_primal_self(ptr %x, double %scale) noinline {
entry:
  %s = call double @__enzyme_test_arg_self(double %scale)
  %vx = load double, ptr %x, align 8
  %prod = fmul double %vx, %s
  store double %prod, ptr %x, align 8
  ret void
}

define void @top_primal_self(ptr %x, double %scale) {
entry:
  call void @callee_primal_self(ptr %x, double %scale)
  %y = load double, ptr %x, align 8
  %sq = fmul double %y, %y
  store double %sq, ptr %x, align 8
  ret void
}

define i32 @main() {
entry:
  %x = alloca double, align 8
  %dx = alloca double, align 8
  %out = alloca double, align 8
  %dout = alloca double, align 8
  %buf1 = alloca double, align 8
  %buf2 = alloca double, align 8

  ; Case 1: out = (2 * x)^2 = 4 * x^2 at x = 3 => dx = 24.0
  store double 3.000000e+00, ptr %x, align 8
  store double 0.000000e+00, ptr %dx, align 8
  store double 0.000000e+00, ptr %out, align 8
  store double 1.000000e+00, ptr %dout, align 8
  call void (...) @__enzyme_autodiff(ptr @top_shadow_only, ptr %x, ptr %dx, ptr %out, ptr %dout)
  %r1 = load double, ptr %dx, align 8
  call i32 (ptr, ...) @printf(ptr @.str, double %r1)

  ; Case 2: out = 3 * (x * w) at x = 4, w = 5 => dx = 15.0
  store double 4.000000e+00, ptr %x, align 8
  store double 0.000000e+00, ptr %dx, align 8
  store double 5.000000e+00, ptr %buf1, align 8
  store double 0.000000e+00, ptr %out, align 8
  store double 1.000000e+00, ptr %dout, align 8
  call void (...) @__enzyme_autodiff(ptr @top_shadow_scalar_tape, ptr %x, ptr %dx, metadata !"enzyme_const", ptr %buf1, ptr %out, ptr %dout)
  %r2 = load double, ptr %dx, align 8
  call i32 (ptr, ...) @printf(ptr @.str, double %r2)

  ; Case 3: out = 2 * (x * w1 * w2) at x = 3, w1 = 5, w2 = 7 => dx = 70.0
  store double 3.000000e+00, ptr %x, align 8
  store double 0.000000e+00, ptr %dx, align 8
  store double 5.000000e+00, ptr %buf1, align 8
  store double 7.000000e+00, ptr %buf2, align 8
  store double 0.000000e+00, ptr %out, align 8
  store double 1.000000e+00, ptr %dout, align 8
  call void (...) @__enzyme_autodiff(ptr @top_shadow_struct_tape, ptr %x, ptr %dx, metadata !"enzyme_const", ptr %buf1, metadata !"enzyme_const", ptr %buf2, ptr %out, ptr %dout)
  %r3 = load double, ptr %dx, align 8
  call i32 (ptr, ...) @printf(ptr @.str, double %r3)

  ; Case 4: out = (4 * x)^2 = 16 * x^2 at x = 2.5 => dx = 80.0
  store double 2.500000e+00, ptr %x, align 8
  store double 0.000000e+00, ptr %dx, align 8
  store double 0.000000e+00, ptr %out, align 8
  store double 1.000000e+00, ptr %dout, align 8
  call void (...) @__enzyme_autodiff(ptr @top_shadow_in_loop, ptr %x, ptr %dx, ptr %out, ptr %dout)
  %r4 = load double, ptr %dx, align 8
  call i32 (ptr, ...) @printf(ptr @.str, double %r4)

  ; Case 5: x_final = (x * scale)^2 at x = 3, scale = 5 => dx = 150.0
  store double 3.000000e+00, ptr %x, align 8
  store double 1.000000e+00, ptr %dx, align 8
  call void (...) @__enzyme_autodiff(ptr @top_primal_tape, ptr %x, ptr %dx, metadata !"enzyme_const", double 5.000000e+00)
  %r5 = load double, ptr %dx, align 8
  call i32 (ptr, ...) @printf(ptr @.str, double %r5)

  ; Case 6: x_final = (x * scale)^2 at x = 2, scale = 7 => dx = 196.0
  store double 2.000000e+00, ptr %x, align 8
  store double 1.000000e+00, ptr %dx, align 8
  call void (...) @__enzyme_autodiff(ptr @top_primal_self, ptr %x, ptr %dx, metadata !"enzyme_const", double 7.000000e+00)
  %r6 = load double, ptr %dx, align 8
  call i32 (ptr, ...) @printf(ptr @.str, double %r6)

  ret i32 0
}

; CHECK-LABEL: define internal void @augmented_callee_shadow_only(
; CHECK-SAME: ptr{{[^,]*}} %[[X1:[^,]+]], ptr{{[^,]*}} %[[DX1:[^,]+]], ptr{{[^,]*}} %[[OUT1:[^,]+]], ptr{{[^,]*}} %[[DOUT1:[^)]+]])
; CHECK-NOT: alloca
; CHECK: ret void

; CHECK-LABEL: define internal void @diffecallee_shadow_only(
; CHECK-SAME: ptr{{[^,]*}} %[[RX1:[^,]+]], ptr{{[^,]*}} %[[RDX1:[^,]+]], ptr{{[^,]*}} %[[ROUT1:[^,]+]], ptr{{[^,]*}} %[[RDOUT1:[^)]+]])
; CHECK: load double, ptr %[[RDX1]]
; CHECK: store double {{.*}}, ptr %[[RDX1]]

; CHECK-LABEL: define internal double @augmented_callee_shadow_scalar_tape(
; CHECK-SAME: ptr{{[^,]*}} %[[X2:[^,]+]], ptr{{[^,]*}} %[[DX2:[^,]+]], ptr{{[^,]*}} %[[BUF2:[^,]+]], ptr{{[^,]*}} %[[OUT2:[^,]+]], ptr{{[^,]*}} %[[DOUT2:[^)]+]])
; CHECK: %[[W2:.*]] = load double, ptr %[[BUF2]]
; CHECK: store double %[[W2]], ptr %[[TAPE_ALLOCA2:[^,]+]]
; CHECK: %[[TAPE_RET2:.*]] = load double, ptr %[[TAPE_ALLOCA2]]
; CHECK: ret double %[[TAPE_RET2]]

; CHECK-LABEL: define internal void @diffecallee_shadow_scalar_tape(
; CHECK-SAME: ptr{{[^,]*}} %[[RX2:[^,]+]], ptr{{[^,]*}} %[[RDX2:[^,]+]], ptr{{[^,]*}} %[[RBUF2:[^,]+]], ptr{{[^,]*}} %[[ROUT2:[^,]+]], ptr{{[^,]*}} %[[RDOUT2:[^,]+]], double %[[TAPE2:[^)]+]])
; CHECK: fmul fast double {{.*}}, %[[TAPE2]]
; CHECK: store double {{.*}}, ptr %[[RDX2]]

; CHECK-LABEL: define internal { double, double } @augmented_callee_shadow_struct_tape(
; CHECK-SAME: ptr{{[^,]*}} %[[X3:[^,]+]], ptr{{[^,]*}} %[[DX3:[^,]+]], ptr{{[^,]*}} %[[B3A:[^,]+]], ptr{{[^,]*}} %[[B3B:[^,]+]], ptr{{[^,]*}} %[[OUT3:[^,]+]], ptr{{[^,]*}} %[[DOUT3:[^)]+]])

; CHECK-LABEL: define internal void @diffecallee_shadow_struct_tape(
; CHECK-SAME: ptr{{[^,]*}} %[[RX3:[^,]+]], ptr{{[^,]*}} %[[RDX3:[^,]+]], ptr{{[^,]*}} %[[RB3A:[^,]+]], ptr{{[^,]*}} %[[RB3B:[^,]+]], ptr{{[^,]*}} %[[ROUT3:[^,]+]], ptr{{[^,]*}} %[[RDOUT3:[^,]+]], { double, double } %[[TAPE3:[^)]+]])
; CHECK: extractvalue { double, double } %[[TAPE3]], 0
; CHECK: extractvalue { double, double } %[[TAPE3]], 1
; CHECK: store double {{.*}}, ptr %[[RDX3]]

; CHECK-LABEL: define internal void @augmented_callee_shadow_in_loop(
; CHECK-SAME: ptr{{[^,]*}} %[[X4:[^,]+]], ptr{{[^,]*}} %[[DX4:[^,]+]], ptr{{[^,]*}} %[[OUT4:[^,]+]], ptr{{[^,]*}} %[[DOUT4:[^)]+]])
; CHECK-NOT: @malloc
; CHECK: ret void

; CHECK-LABEL: define internal void @diffecallee_shadow_in_loop(
; CHECK-SAME: ptr{{[^,]*}} %[[RX4:[^,]+]], ptr{{[^,]*}} %[[RDX4:[^,]+]], ptr{{[^,]*}} %[[ROUT4:[^,]+]], ptr{{[^,]*}} %[[RDOUT4:[^)]+]])
; CHECK: store ptr %[[RDX4]], ptr

; CHECK-LABEL: define internal void @augmented_callee_primal_tape(
; CHECK-SAME: ptr{{[^,]*}} %[[X5:[^,]+]], ptr{{[^,]*}} %[[DX5:[^,]+]], double %[[SCALE5:[^)]+]])

; CHECK-LABEL: define internal void @diffecallee_primal_tape(
; CHECK-SAME: ptr{{[^,]*}} %[[RX5:[^,]+]], ptr{{[^,]*}} %[[RDX5:[^,]+]], double %[[RSCALE5:[^)]+]])
; CHECK: %[[PREV5:.*]] = load double, ptr %[[RDX5]]
; CHECK-NEXT: %[[SCALED5:.*]] = fmul {{.*}}double %[[PREV5]], %[[RSCALE5]]
; CHECK-NEXT: store double %[[SCALED5]], ptr %[[RDX5]]

; CHECK-LABEL: define internal void @augmented_callee_primal_self(
; CHECK-SAME: ptr{{[^,]*}} %[[X6:[^,]+]], ptr{{[^,]*}} %[[DX6:[^,]+]], double %[[SCALE6:[^)]+]])

; CHECK-LABEL: define internal void @diffecallee_primal_self(
; CHECK-SAME: ptr{{[^,]*}} %[[RX6:[^,]+]], ptr{{[^,]*}} %[[RDX6:[^,]+]], double %[[RSCALE6:[^)]+]])
; CHECK: fmul fast double {{.*}}, %[[RSCALE6]]
; CHECK: store double {{.*}}, ptr %[[RDX6]]

; EVAL: 24.0
; EVAL-NEXT: 15.0
; EVAL-NEXT: 70.0
; EVAL-NEXT: 80.0
; EVAL-NEXT: 150.0
; EVAL-NEXT: 196.0
