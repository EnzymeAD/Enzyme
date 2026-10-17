; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; The load and the store are in two loops in the two branches of an if, so
; neither loop header dominates the other. Their addresses use 32-bit induction
; variables sign-extended to 64 bits, so the address SCEVs are not AddRecs
; themselves but contain AddRecs of the two loops:
;
;   for (i = 0; i < N; i++) {
;     if (i & 1)
;       for (j = 0; j != M; j++) s += A[(long)j] * A[(long)j];
;     else
;       for (k = 0; k != M; k++) A[(long)M - 1 - (long)k] = 0.0;
;   }
;
; The difference of the two addresses then contains the terms
; 8 * sext({0,+,1}<%load.loop>) and 8 * sext({0,+,1}<%store.loop>), and to sort
; them ScalarEvolution compares the two AddRecs. overwritesToMemoryReadByLoop
; must not subtract these SCEVs, else an assertions build of LLVM fails with
; "No dominance between recurrences used by one SCEV?".

declare double @__enzyme_autodiff(ptr, ...)

define double @f(ptr %A, i64 %N, i32 %M) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %s = phi double [ 0.000000e+00, %entry ], [ %s.out, %latch ]
  %odd = and i64 %i, 1
  %isodd = icmp ne i64 %odd, 0
  br i1 %isodd, label %load.loop, label %store.loop

load.loop:
  %j = phi i32 [ 0, %outer ], [ %j.next, %load.loop ]
  %sl = phi double [ %s, %outer ], [ %sl.next, %load.loop ]
  %j.ext = sext i32 %j to i64
  %ldp = getelementptr inbounds double, ptr %A, i64 %j.ext
  %v = load double, ptr %ldp, align 8
  %sq = fmul double %v, %v
  %sl.next = fadd double %sl, %sq
  %j.next = add i32 %j, 1
  %j.cmp = icmp ne i32 %j.next, %M
  br i1 %j.cmp, label %load.loop, label %latch

store.loop:
  %k = phi i32 [ 0, %outer ], [ %k.next, %store.loop ]
  %k.ext = sext i32 %k to i64
  %M.ext = sext i32 %M to i64
  %last = add i64 %M.ext, -1
  %idx = sub i64 %last, %k.ext
  %stp = getelementptr inbounds double, ptr %A, i64 %idx
  store double 0.000000e+00, ptr %stp, align 8
  %k.next = add i32 %k, 1
  %k.cmp = icmp ne i32 %k.next, %M
  br i1 %k.cmp, label %store.loop, label %latch

latch:
  %s.out = phi double [ %sl.next, %load.loop ], [ %s, %store.loop ]
  %i.next = add nuw nsw i64 %i, 1
  %i.cmp = icmp ne i64 %i.next, %N
  br i1 %i.cmp, label %outer, label %exit

exit:
  ret double %s.out
}

define double @df(ptr %A, ptr %dA, i64 %N, i32 %M) {
entry:
  %r = call double (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %A, ptr %dA, i64 %N, i32 %M)
  ret double %r
}

; The store of a later outer iteration may overwrite what the load read, so %v
; is cached in the forward pass and read back in the reverse pass.

; CHECK: define internal void @diffef(ptr {{.*}}%A, ptr {{.*}}%"A'", i64 %N, i32 %M, double %differeturn)
; CHECK: load.loop:
; CHECK:   %v = load double, ptr %ldp
; CHECK:   store double %v, ptr
; CHECK: invertload.loop:
; CHECK:   load double, ptr {{.*}}, !invariant.group
