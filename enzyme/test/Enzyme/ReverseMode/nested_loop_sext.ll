; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,adce)" -S | FileCheck %s

define void @nested(float* %x, i32 %outer, i32 %inner) {
entry:
  %cmp.out = icmp slt i32 %outer, 1
  br i1 %cmp.out, label %exit, label %outer.preheader

outer.preheader:
  br label %outer.loop

outer.loop:
  %iv.out = phi i32 [ 1, %outer.preheader ], [ %iv.out.next, %outer.loop.latch ]
  
  ; Inner loop guard
  %cmp.in = icmp slt i32 %inner, 1
  br i1 %cmp.in, label %outer.loop.latch, label %inner.preheader

inner.preheader:
  br label %inner.loop

inner.loop:
  %iv.in = phi i32 [ 1, %inner.preheader ], [ %iv.in.next, %inner.loop ]
  
  %idx = getelementptr inbounds float, float* %x, i32 %iv.in
  %val = load float, float* %idx, align 4
  %val2 = fmul float %val, %val
  store float %val2, float* %idx, align 4
  
  %iv.in.next = add i32 %iv.in, 1
  %cmp.in.loop = icmp eq i32 %iv.in, %inner
  br i1 %cmp.in.loop, label %outer.loop.latch, label %inner.loop

outer.loop.latch:
  %iv.out.next = add i32 %iv.out, 1
  %cmp.out.loop = icmp eq i32 %iv.out, %outer
  br i1 %cmp.out.loop, label %exit, label %outer.loop

exit:
  ret void
}

declare void @__enzyme_autodiff(i8*, float*, float*, i32, i32)

define void @diffe_nested(float* %x, float* %xp, i32 %outer, i32 %inner) {
entry:
  call void @__enzyme_autodiff(i8* bitcast (void (float*, i32, i32)* @nested to i8*), float* %x, float* %xp, i32 %outer, i32 %inner)
  ret void
}

; CHECK: define internal void @diffenested
; CHECK:   %[[sub:.*]] = add i32 %inner, -1
; CHECK:   %{{.*}} = sext i32 %[[sub]] to i64
