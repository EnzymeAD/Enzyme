; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

declare i1 @cmp()

define void @f(double* %y) {
entry:
  br label %L44.i

L44.i: 
  %iv = phi i64 [ %iv.next, %L44.i ], [ 0, %entry ]
  %iv.next = add nuw nsw i64 %iv, 1
  %a16 = getelementptr inbounds double, double* %y, i64 %iv
  store double 3.141592e+00, double* %a16, align 8
  %value_phi24.i = call i1 @cmp() inaccessiblememonly "enzyme_inactive"
  br i1 %value_phi24.i, label %julia_f_mut__821_inner.exit.loopexit, label %L44.i

julia_f_mut__821_inner.exit.loopexit:             ; preds = %L44.i
  ret void
}

declare void @__enzyme_autodiff(...)

define void @test(double* %y, double* %dy) {
entry:
  call void (...) @__enzyme_autodiff(void (double*)* @f, metadata !"enzyme_dup", double* %y, double* %dy)
  ret void
}

; CHECK: define internal void @diffef(double* %y, double* %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"iv1'ac" = alloca i64, align 8
; CHECK-NEXT:   %loopLimit_cache = alloca i64, align 8
; CHECK-NEXT:   br label %L44.i

; CHECK: L44.i:                                            ; preds = %L44.i, %entry
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %L44.i ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %"a16'ipg" = getelementptr inbounds double, double* %"y'", i64 %iv1
; CHECK-NEXT:   %a16 = getelementptr inbounds double, double* %y, i64 %iv1
; CHECK-NEXT:   store double 0x400921FAFC8B007A, double* %a16, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %value_phi24.i = call i1 @cmp() #2
; CHECK-NEXT:   br i1 %value_phi24.i, label %julia_f_mut__821_inner.exit.loopexit, label %L44.i

; CHECK: julia_f_mut__821_inner.exit.loopexit:             ; preds = %L44.i
; CHECK-NEXT:   %0 = phi i64 [ %iv1, %L44.i ]
; CHECK-NEXT:   store i64 %0, i64* %loopLimit_cache, align 8, !invariant.group !5
; CHECK-NEXT:   br label %invertjulia_f_mut__821_inner.exit.loopexit

; CHECK: invertentry:                                      ; preds = %invertL44.i
; CHECK-NEXT:   ret void

; CHECK: invertL44.i:  
; CHECK-NEXT:   %1 = load i64, i64* %"iv1'ac", align 4
; CHECK-NEXT:   %"a16'ipg_unwrap" = getelementptr inbounds double, double* %"y'", i64 %1
; CHECK-NEXT:   store double 0.000000e+00, double* %"a16'ipg_unwrap", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %2 = load i64, i64* %"iv1'ac", align 4
; CHECK-NEXT:   %3 = icmp eq i64 %2, 0
; CHECK-NEXT:   %4 = xor i1 %3, true
; CHECK-NEXT:   br i1 %3, label %invertentry, label %incinvertL44.i

; CHECK: incinvertL44.i:                                   ; preds = %invertL44.i
; CHECK-NEXT:   %5 = load i64, i64* %"iv1'ac", align 4
; CHECK-NEXT:   %6 = add nsw i64 %5, -1
; CHECK-NEXT:   store i64 %6, i64* %"iv1'ac", align 4
; CHECK-NEXT:   br label %invertL44.i

; CHECK: invertjulia_f_mut__821_inner.exit.loopexit:       ; preds = %julia_f_mut__821_inner.exit.loopexit
; CHECK-NEXT:   %7 = load i64, i64* %loopLimit_cache, align 8, !invariant.group !5
; CHECK-NEXT:   br label %mergeinvertL44.i_julia_f_mut__821_inner.exit.loopexit

; CHECK: mergeinvertL44.i_julia_f_mut__821_inner.exit.loopexit: ; preds = %invertjulia_f_mut__821_inner.exit.loopexit
; CHECK-NEXT:   store i64 %7, i64* %"iv1'ac", align 4
; CHECK-NEXT:   br label %invertL44.i
; CHECK-NEXT: }
