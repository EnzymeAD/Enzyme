; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme-strict-aliasing=0 -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme-strict-aliasing=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

declare void @__enzyme_autodiff(...)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

define void @square(i1 %c, i8* %0, i8* %1) {
entry:
  br i1 %c, label %run, label %end

run:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double, [-1,40]:Float@double, [-1,48]:Float@double, [-1,56]:Float@double, [-1,64]:Float@double, [-1,72]:Float@double}" %0, i8* %1, i64 "enzyme_type"="{[-1]:Integer}" 80, i1 false)
  br label %end

end:
  ret void
}

define void @dsquare(i8* %x, i8* %dx, i8* %y, i8* %dy) {
entry:
  tail call void (...) @__enzyme_autodiff(void (i1, i8*, i8*)* nonnull @square, i1 false, metadata !"enzyme_dup", i8* %x, i8* %dx, metadata !"enzyme_dup", i8* %y, i8* %dy)
  ret void
}

; CHECK: define internal void @diffesquare(i1 %c, i8* %0, i8* %"'", i8* %1, i8* %"'1")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 %c, label %run, label %invertentry

; CHECK: run: 
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double, [-1,40]:Float@double, [-1,48]:Float@double, [-1,56]:Float@double, [-1,64]:Float@double, [-1,72]:Float@double}" %0, i8* %1, i64 "enzyme_type"="{[-1]:Integer}" 80, i1 false)
; CHECK-NEXT:   %2 = bitcast i8* %"'" to double*
; CHECK-NEXT:   %3 = bitcast i8* %"'1" to double*
; CHECK-NEXT:   br label %for.body.i

; CHECK: invertentry:   
; CHECK-NEXT:   ret void

; CHECK: for.body.i: 
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %run ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %2, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load double, double* %dst.i.i, align 1
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %3, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i, align 1
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store double %[[i4]], double* %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %[[i5:.+]] = icmp eq i64 10, %idx.next.i
; CHECK-NEXT:   br i1 %[[i5]], label %invertentry, label %for.body.i
; CHECK-NEXT: }
