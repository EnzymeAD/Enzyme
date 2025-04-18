; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -gt 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -loop-deletion -simplifycfg -correlated-propagation -adce -instsimplify -S | FileCheck %s; fi
; RUN: if [ %llvmver -gt 11 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,loop(loop-deletion),%simplifycfg,correlated-propagation,adce,instsimplify)" -enzyme-preopt=false -S | FileCheck %s; fi

; Function Attrs: nounwind
declare void @__enzyme_autodiff(i8*, ...)

define void @test_derivative(double addrspace(1)* %in, double addrspace(1)* %din) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double addrspace(1)*)* @g to i8*), double addrspace(1)* %in, double addrspace(1)* %din)
  ret void
}

define double @g(double addrspace(1)* %in) {
top:
  br label %L1

L1:                                              ; preds = %L50, %top
  %i = phi i64 [ 0, %top ], [ %ince, %L1e ]
  %sum0 = phi double [ 0.000000e+00, %top ], [ %add, %L1e ]
  br label %L50

L50:                                              ; preds = %L50, %top
  %value_phi = phi i64 [ 0, %L1 ], [ %inc, %L50 ]
  %sum = phi double [ %sum0, %L1 ], [ %add, %L50 ]
  %gep = getelementptr inbounds double, double addrspace(1)* %in, i64 %i
  %ld = load double, double addrspace(1)* %gep, align 8
  %sq = fmul double %ld, %ld
  %add = fadd double %sum, %sq
  %.not.not = icmp eq i64 %value_phi, 5
  %inc = add nuw nsw i64 %value_phi, 1
  br i1 %.not.not, label %L1e, label %L50

L1e:
  %ince = add nuw nsw i64 %i, 1
  %l1e = icmp eq i64 %ince, 10
  br i1 %l1e, label %exit, label %L1

exit:                                             ; preds = %L50
  store double 0.000000e+00, double addrspace(1)* %in, align 8
  ret double %add
}

; CHECK: define internal void @diffeg(double addrspace(1)* %in, double addrspace(1)* %"in'", double %differeturn)
; CHECK-NEXT: top:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %ld_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %0 = bitcast double addrspace(1)* %in to i8 addrspace(1)*
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p1i8.i64(i8* nonnull align 8 %malloccall, i8 addrspace(1)* nonnull align 8 %0, i64 80, i1 false)
; CHECK-NEXT:   store double 0.000000e+00, double addrspace(1)* %in, align 8
; CHECK-NEXT:   store double 0.000000e+00, double addrspace(1)* %"in'", align 8
; CHECK-NEXT:   br label %invertL1e

; CHECK: inverttop:                                        ; preds = %invertL1
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertL1:                                         ; preds = %invertL50
; CHECK-NEXT:   %1 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %1, label %inverttop, label %incinvertL1

; CHECK: incinvertL1:                                      ; preds = %invertL1
; CHECK-NEXT:   %2 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertL1e

; CHECK: invertL50:                                        ; preds = %invertL1e, %incinvertL50
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 5, %invertL1e ], [ %[[aa:.+]], %incinvertL50 ]
; CHECK-NEXT:   %[[a4:.+]] = getelementptr inbounds double, double* %ld_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[a5:.+]] = load double, double* %[[a4]], align 8
; CHECK-NEXT:   %[[m0diffeld:.+]] = fmul fast double %differeturn, %[[a5]]
; CHECK-NEXT:   %[[m1diffeld:.+]] = fmul fast double %differeturn, %[[a5]]
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %[[m0diffeld]], %[[m1diffeld]]
; CHECK-NEXT:   %"gep'ipg_unwrap" = getelementptr inbounds double, double addrspace(1)* %"in'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[i6:.+]] = load double, double addrspace(1)* %"gep'ipg_unwrap"
; CHECK-NEXT:   %[[i7:.+]] = fadd fast double %[[i6]], %[[i4]]
; CHECK-NEXT:   store double %[[i7]], double addrspace(1)* %"gep'ipg_unwrap"
; CHECK-NEXT:   %[[i6:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[i6]], label %invertL1, label %incinvertL50

; CHECK: incinvertL50:                                     ; preds = %invertL50
; CHECK-NEXT:   %[[aa]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertL50

; CHECK: invertL1e:                                        ; preds = %top, %incinvertL1
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 9, %top ], [ %2, %incinvertL1 ]
; CHECK-NEXT:   br label %invertL50
; CHECK-NEXT: }
