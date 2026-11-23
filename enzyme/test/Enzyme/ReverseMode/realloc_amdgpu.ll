; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -S | FileCheck %s --implicit-check-not=@realloc; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify),verify" -S | FileCheck %s --implicit-check-not=@realloc
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify),verify" -S | sed '/^target triple/d;/^target datalayout/d' | %lli -
; REQUIRES: amdgpu-registered-target

; AMDGPU device code has no realloc: the cache of a loop whose trip count is not
; known on entry grows by malloc and a copy, and frees the buffer it replaces.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@xs = global [6 x double] [double 1.500000e+00, double 1.500000e+00, double 1.500000e+00, double 1.500000e+00, double 1.500000e+00, double 1.500000e+00]
@dxs = global [6 x double] zeroinitializer

define double @f(ptr %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %p
  store double 0.000000e+00, ptr %p
  %sq = fmul double %v, %v
  %acc.next = fadd double %acc, %sq
  %i.next = add nuw i64 %i, 1
  %c = fcmp ogt double %acc.next, 1.000000e+01
  br i1 %c, label %exit, label %loop

exit:
  ret double %acc.next
}

define void @df(ptr %x, ptr %dx, i64 %n) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, i64 %n)
  ret void
}

declare double @__enzyme_autodiff(...)

define i32 @main() {
entry:
  ; At x[i] = 1.5 the loop runs five steps (5 * 2.25 > 10), so the cache grows
  ; at sizes 1, 3 and 5 and frees a replaced buffer twice; dx[i] = 2 * x[i] = 3
  ; for i < 5, and dx[5] stays 0 since the loop never reads x[5].
  call void @df(ptr @xs, ptr @dxs, i64 6)
  %p0 = getelementptr inbounds [6 x double], ptr @dxs, i64 0, i64 0
  %d0 = load double, ptr %p0
  %p4 = getelementptr inbounds [6 x double], ptr @dxs, i64 0, i64 4
  %d4 = load double, ptr %p4
  %p5 = getelementptr inbounds [6 x double], ptr @dxs, i64 0, i64 5
  %d5 = load double, ptr %p5
  %ok0 = fcmp oeq double %d0, 3.000000e+00
  %ok4 = fcmp oeq double %d4, 3.000000e+00
  %ok5 = fcmp oeq double %d5, 0.000000e+00
  %ok04 = and i1 %ok0, %ok4
  %ok = and i1 %ok04, %ok5
  %rc = select i1 %ok, i32 0, i32 1
  ret i32 %rc
}

; CHECK-LABEL: define internal void @diffef(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n, double %differeturn)
; CHECK: loop:
; CHECK-NEXT:   %[[CACHE:.+]] = phi ptr [ null, %entry ], [ %[[NEXT_CACHE:.+]], %__enzyme_exponentialallocation.exit ]
; CHECK:   br i1 %{{.+}}, label %grow.i, label %__enzyme_exponentialallocation.exit
; CHECK: grow.i:
; CHECK-NEXT:   %[[LZ:.+]] = call i64 @llvm.ctlz.i64(i64 %iv.next, i1 true)
; CHECK-NEXT:   %[[LOG:.+]] = sub nuw nsw i64 64, %[[LZ]]
; CHECK-NEXT:   %[[SIZE:.+]] = shl i64 8, %[[LOG]]
; CHECK-NEXT:   %[[HALF:.+]] = lshr i64 %[[SIZE]], 1
; CHECK-NEXT:   %[[FIRST:.+]] = icmp eq i64 %iv.next, 1
; CHECK-NEXT:   %[[PREV:.+]] = select i1 %[[FIRST]], i64 0, i64 %[[HALF]]
; CHECK-NEXT:   %[[NEWBUF:.+]] = call noalias nonnull ptr @malloc(i64 %[[SIZE]])
; CHECK-NEXT:   %[[TAILSIZE:.+]] = sub i64 %[[SIZE]], %[[PREV]]
; CHECK-NEXT:   %[[TAIL:.+]] = getelementptr inbounds i8, ptr %[[NEWBUF]], i64 %[[PREV]]
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr %[[TAIL]], i8 0, i64 %[[TAILSIZE]], i1 false)
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %[[NEWBUF]], ptr %[[CACHE]], i64 %[[PREV]], i1 false)
; CHECK-NEXT:   br i1 %[[FIRST]], label %afterfree.i, label %freeprev.i
; CHECK: freeprev.i:
; CHECK-NEXT:   call void @free(ptr nonnull %[[CACHE]])
; CHECK-NEXT:   br label %afterfree.i
; CHECK: afterfree.i:
; CHECK-NEXT:   br label %__enzyme_exponentialallocation.exit
; CHECK: __enzyme_exponentialallocation.exit:
; CHECK-NEXT:   %[[NEXT_CACHE]] = phi ptr [ %[[NEWBUF]], %afterfree.i ], [ %[[CACHE]], %loop ]
; CHECK-NEXT:   %p = getelementptr inbounds double, ptr %x, i64 %iv
; CHECK-NEXT:   %v = load double, ptr %p, align 8
; CHECK-NEXT:   %[[SLOT:.+]] = getelementptr inbounds {{\[8 x i8\]|double}}, ptr %[[NEXT_CACHE]], i64 %iv
; CHECK-NEXT:   store double %v, ptr %[[SLOT]], align 8, !invariant.group
; CHECK: invertentry:
; CHECK-NEXT:   tail call void @free(ptr nonnull %[[NEXT_CACHE]])
; CHECK-NEXT:   ret void
; CHECK: invertloop:
; CHECK:   %[[RSLOT:.+]] = getelementptr inbounds {{\[8 x i8\]|double}}, ptr %[[NEXT_CACHE]], i64 %"iv'ac.0"
; CHECK-NEXT:   %{{.+}} = load double, ptr %[[RSLOT]], align 8, !invariant.group
