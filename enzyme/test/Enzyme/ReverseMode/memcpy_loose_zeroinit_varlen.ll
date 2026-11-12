; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme-loose-types=1 -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-loose-types=1 -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

@g = internal constant [10 x i8] zeroinitializer

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @__enzyme_autodiff(...)

define void @zeroinit_copy(ptr %dst, i64 %n) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr @g, i64 %n, i1 false)
  ret void
}

define void @caller_zeroinit(ptr %dst, ptr %ddst, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @zeroinit_copy, metadata !"enzyme_dup", ptr %dst, ptr %ddst, i64 %n)
  ret void
}

define void @outofrange_copy(ptr %dst, ptr %src, i64 %n) {
entry:
  %p = getelementptr inbounds i8, ptr %src, i64 8
  store double 2.000000e+00, ptr %p, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  ret void
}

define void @caller_outofrange(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @outofrange_copy, metadata !"enzyme_dup", ptr %dst, ptr %ddst, metadata !"enzyme_dup", ptr %src, ptr %dsrc, i64 %n)
  ret void
}

; CHECK: define internal void @diffezeroinit_copy(ptr {{.*}}%dst, ptr {{.*}}%"dst'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %"dst'", ptr align 1 @g, i64 %n, i1 false)
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr @g, i64 %n, i1 false)
; CHECK-NEXT:   ret void

; CHECK: define internal void @diffeoutofrange_copy(ptr {{.*}}%dst, ptr {{.*}}%"dst'", ptr {{.*}}%src, ptr {{.*}}%"src'", i64 %n)
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
; CHECK:   %0 = udiv i64 %n, 8
; CHECK:   br i1 {{.*}}, label %__enzyme_memcpyadd_doubleda1sa1.exit, label %for.body.i
