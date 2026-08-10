; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; GradientUtils::legalRecompute maps `origStart` and the load back into the
; original function (via isOriginal) and checks OrigDT, but used to hand
; CacheUtility's `LI` -- LoopInfo over the *new* function -- to
; allInstructionsBetween, which asserts
;   GenericLoopInfo.h: verifyBlockNumberEpoch:
;   `ParentPtr == BBParent && "loop info queried with block of other function"'
;
; The inner loop's trip count is an i32 load that the loop's own double store
; may clobber, so it is uncacheable and has to be recomputed; that recompute is
; what reaches legalRecompute with a forward-mode builder. @f calling @inner
; keeps the load one subfunction level below the autodiff entry point, and the
; opaque call after the loop is required -- without it the load is not
; recomputed here and the assertion does not fire.

declare void @sink(ptr, ptr) nofree "enzyme_inactive"

define internal void @inner(ptr %out, ptr %n, ptr %co, i1 %rep) {
entry:
  br label %outer

outer:                                            ; preds = %latch, %entry
  %cnt = load i32, ptr %n, align 4
  %cnt64 = zext i32 %cnt to i64
  br label %body

body:                                             ; preds = %body, %outer
  %i = phi i64 [ 0, %outer ], [ %i.next, %body ]
  %co.i = getelementptr inbounds double, ptr %co, i64 %i
  %c0 = load double, ptr %co.i, align 8
  %acc = load double, ptr %out, align 8
  %mul = fmul double %c0, %acc
  store double %mul, ptr %out, align 8
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i, %cnt64
  br i1 %done, label %latch, label %body

latch:                                            ; preds = %body
  br i1 %rep, label %outer, label %exit

exit:                                             ; preds = %latch
  call void @sink(ptr %out, ptr %co)
  ret void
}

define void @f(ptr %out, ptr %n, ptr %co) {
entry:
  call void @inner(ptr %out, ptr %n, ptr %co, i1 false)
  ret void
}

define void @dtarget(ptr %out, ptr %dout, ptr %n, ptr %co, ptr %dco) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, ptr %out, ptr %dout, ptr @enzyme_const, ptr %n, ptr %co, ptr %dco)
  ret void
}

declare void @__enzyme_autodiff(...)

@enzyme_const = external global ptr

; CHECK: define internal void @diffef(ptr %out, ptr %"out'", ptr{{.*}} %n, ptr %co, ptr %"co'")
; CHECK: call void @diffeinner(ptr %out, ptr %"out'", ptr{{.*}} %n, ptr %co, ptr %"co'", i1 false)

; CHECK: define internal void @diffeinner(ptr %out, ptr %"out'", ptr{{.*}} %n, ptr %co, ptr %"co'", i1 %rep)
; the uncacheable trip count is recomputed, not cached
; CHECK: %cnt_unwrap = load i32, ptr %n
; CHECK: %cnt64_unwrap = zext i32 %cnt_unwrap to i64
