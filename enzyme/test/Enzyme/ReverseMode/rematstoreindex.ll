; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; %arr is rematerialized in the reverse pass by replaying its stores. The
; address of one store uses the index %i, which is loaded from a local that an
; opaque call fills in (like a Fortran READ) and that is overwritten later. So
; %i can be neither reloaded nor recomputed: it has to be cached for the
; replayed store. It used to be dropped, which gave
; "Illegal replace ficticious phi".

declare void @read_int(ptr) #0
declare void @__enzyme_autodiff(...)

define void @f(ptr %x, i64 %n) {
entry:
  %idx = alloca i32, align 4
  %arr = alloca [2 x i32], align 4
  store i32 0, ptr %arr, align 4
  %arr1 = getelementptr i32, ptr %arr, i64 1
  store i32 0, ptr %arr1, align 4
  call void @read_int(ptr %idx)
  %i = load i32, ptr %idx, align 4
  %ie = sext i32 %i to i64
  %slot = getelementptr i32, ptr %arr, i64 %ie
  store i32 1, ptr %slot, align 4
  store i32 7, ptr %idx, align 4
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %k1, %loop ]
  %ap = getelementptr i32, ptr %arr, i64 %k
  %v = load i32, ptr %ap, align 4
  %ve = sext i32 %v to i64
  %xp = getelementptr double, ptr %x, i64 %ve
  %xv = load double, ptr %xp, align 8
  %sq = fmul double %xv, %xv
  store double %sq, ptr %xp, align 8
  %k1 = add nuw i64 %k, 1
  %c = icmp eq i64 %k1, %n
  br i1 %c, label %exit, label %loop

exit:
  ret void
}

define void @g(ptr %x, i64 %n) {
  call void @f(ptr %x, i64 %n)
  %xv = load double, ptr %x, align 8
  %sq = fmul double %xv, %xv
  store double %sq, ptr %x, align 8
  ret void
}

define void @df(ptr %x, ptr %dx, i64 %n) {
  call void (...) @__enzyme_autodiff(ptr @g, ptr %x, ptr %dx, i64 %n)
  ret void
}
attributes #0 = { nofree "enzyme_inactive" "enzyme_no_escaping_allocation" }

; CHECK: define internal { i32, ptr } @augmented_f(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n)
; CHECK:   call void @read_int(ptr %idx)
; CHECK-NEXT:   %i = load i32, ptr %idx, align 4
; CHECK-NEXT:   store i32 %i, ptr %0, align 4

; CHECK: define internal void @diffef(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n, { i32, ptr } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %arr = alloca [2 x i32], i64 1, align 4
; CHECK:   store i32 0, ptr %arr, align 4
; CHECK-NEXT:   %arr1 = getelementptr i32, ptr %arr, i64 1
; CHECK-NEXT:   store i32 0, ptr %arr1, align 4
; CHECK-NEXT:   %i = extractvalue { i32, ptr } %tapeArg, 0
; CHECK-NEXT:   %ie = sext i32 %i to i64
; CHECK-NEXT:   %slot = getelementptr i32, ptr %arr, i64 %ie
; CHECK-NEXT:   store i32 1, ptr %slot, align 4
