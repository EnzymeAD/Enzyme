; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; A memcpy at width 2 whose source is a julia.gc_loaded call. When the memcpy is
; visited, the source's shadow is still a placeholder phi, and looking up the
; first element's shadow in the reverse pass (unwrapM) replaces it with the real
; shadow. The second element must then be taken from the replacement.
; Reduced from Julia's IR for `for p in v; e += energy(p); end` (the loop's first
; iteration peeled off), differentiated with a width of 2.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"

define internal double @energy(ptr nocapture noundef nonnull readonly align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}" %p) #0 {
top:
  %w_ptr = getelementptr inbounds i8, ptr %p, i64 8
  %w = load double, ptr %w_ptr, align 8
  %w2 = fmul double %w, %w
  %x = load double, ptr %p, align 8
  %e = fmul double %x, %w2
  ret double %e
}

define double @f(ptr nocapture noundef nonnull readonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" %v) {
top:
  %slot = alloca [2 x i64], align 8
  %size_ptr = getelementptr inbounds i8, ptr %v, i64 16
  %size = load i64, ptr %size_ptr, align 8
  %empty = icmp eq i64 %size, 0
  br i1 %empty, label %exit, label %first

first:
  %mem_ptr = getelementptr inbounds { ptr, ptr }, ptr %v, i64 0, i32 1
  %mem = load ptr, ptr %mem_ptr, align 8
  %data = load ptr, ptr %v, align 8
  %p1 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" ptr @julia.gc_loaded(ptr %mem, ptr %data)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %slot, ptr noundef nonnull align 8 dereferenceable(16) %p1, i64 16, i1 false)
  %e1 = call double @energy(ptr nocapture readonly %slot)
  %more = icmp ugt i64 %size, 1
  br i1 %more, label %loop, label %exit

loop:
  %i = phi i64 [ 1, %first ], [ %i.next, %loop ]
  %acc = phi double [ %e1, %first ], [ %acc.next, %loop ]
  %data.l = load ptr, ptr %v, align 8
  %mem.l = load ptr, ptr %mem_ptr, align 8
  %base = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" ptr @julia.gc_loaded(ptr %mem.l, ptr %data.l)
  %off = shl i64 %i, 4
  %pi = getelementptr inbounds i8, ptr %base, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %slot, ptr noundef align 8 dereferenceable(16) %pi, i64 16, i1 false)
  %e = call double @energy(ptr nocapture readonly %slot)
  %acc.next = fadd double %acc, %e
  %i.next = add i64 %i, 1
  %cont = icmp ult i64 %i.next, %size
  br i1 %cont, label %loop, label %exit

exit:
  %r = phi double [ 0.000000e+00, %top ], [ %e1, %first ], [ %acc.next, %loop ]
  ret double %r
}

declare noundef nonnull ptr @julia.gc_loaded(ptr nocapture noundef nonnull readnone, ptr noundef nonnull readnone) #1

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

declare void @__enzyme_autodiff(ptr, ...)

define void @test_derivative(ptr %v, ptr %dv1, ptr %dv2) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %v, ptr %dv1, ptr %dv2)
  ret void
}

attributes #0 = { noinline nounwind memory(argmem: read) }
attributes #1 = { nofree norecurse nosync nounwind speculatable willreturn memory(argmem: read) "enzyme_nocache" "enzyme_shouldrecompute" }

; CHECK-LABEL: define internal void @diffe2f(
; CHECK: invertfirst:
; CHECK: %[[LANE1:[0-9]+]] = call ptr @julia.gc_loaded(ptr %"mem'il_phi_unwrap{{[0-9]+}}", ptr %"data'il_phi_unwrap{{[0-9]+}}")
; CHECK-NEXT: %[[LANE0:[0-9]+]] = call ptr @julia.gc_loaded(ptr %"mem'il_phi_unwrap", ptr %"data'il_phi_unwrap")
; CHECK: getelementptr inbounds double, ptr %"slot'ipa", i64
; CHECK: getelementptr inbounds double, ptr %[[LANE0]], i64
; CHECK: getelementptr inbounds double, ptr %"slot'ipa1", i64
; CHECK: getelementptr inbounds double, ptr %[[LANE1]], i64
