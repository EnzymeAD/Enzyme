; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=0 -enzyme -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -simplifycfg -adce -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=0  -passes="enzyme,function(mem2reg,early-cse,%simplifycfg,instsimplify,correlated-propagation,%simplifycfg,adce)" -S -enzyme-detect-readthrow=0 | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

; Function Attrs: norecurse nounwind uwtable
define double @alldiv(double* %a, i1 %cmp, i32 %val) {
entry:
  br i1 %cmp, label %mid, label %fin

mid:
  ; Keep the inner predicate available on every path to the merge.
  %c2 = icmp eq i32 %val, 17
  %c1 = icmp eq i32 %val, 13
  br i1 %c1, label %bdef, label %mid2

mid2:
  br i1 %c2, label %b1, label %b2

b1: 
  %g1 = getelementptr inbounds double, double* %a, i32 32
  %l1 = load double, double* %g1, align 8
  br label %end

b2: 
  %g2 = getelementptr inbounds double, double* %a, i32 64
  %l2 = load double, double* %g2, align 8
  br label %end

bdef: 
  %g3 = getelementptr inbounds double, double* %a, i32 128
  %l3 = load double, double* %g3, align 8
  br label %end

end:
  %p = phi double [ %l1, %b1 ], [ %l2, %b2 ], [ %l3, %bdef ]
  %sq = fmul double %p, %p
  br label %fin

fin:
  %res = phi double [ 0.000000e+00, %entry ], [ %sq, %end ]
  ret double %res
}

define void @main(double* %a, double* %da, i1 %N, i32 %N2) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, i1, i32)* @alldiv to i8*), double* nonnull %a, double* nonnull %da, i1 %N, i32 %N2)
  ret void
}

; CHECK-LABEL: define internal void @diffealldiv(
; CHECK: entry:
; CHECK: br i1 %cmp, label %invertend, label %invertentry

; CHECK: invertb1:
; CHECK: getelementptr inbounds double, {{.*}} %"a'", i32 32
; CHECK: store double
; CHECK: invertb2:
; CHECK: getelementptr inbounds double, {{.*}} %"a'", i32 64
; CHECK: store double
; CHECK: invertbdef:
; CHECK: getelementptr inbounds double, {{.*}} %"a'", i32 128
; CHECK: store double

; CHECK: invertend:
; CHECK: %c1_unwrap = icmp eq i32 %val, 13
; CHECK: %c2_unwrap = icmp eq i32 %val, 17
; CHECK: br i1 %c1_unwrap, label %[[DEF:.+]], label %invertend_phisplt
; CHECK: invertend_phisplt:
; CHECK-NEXT: br i1 %c2_unwrap, label %[[ONE:.+]], label %[[TWO:.+]]
; CHECK: [[ONE]]:
; CHECK: %l1_unwrap = load double
; CHECK: [[TWO]]:
; CHECK: %l2_unwrap = load double
; CHECK: [[DEF]]:
; CHECK: %l3_unwrap = load double

; CHECK: invertend_phimerge:
; CHECK: phi {{(fast )?}}double
; CHECK: %anot1_ = xor i1 %c1_unwrap, true
; CHECK-NEXT: %andVal0 = select i1 %anot1_, i1 %c2_unwrap, i1 false
; CHECK-NEXT: %bnot1_ = xor i1 %c2_unwrap, true
; CHECK-NEXT: %andVal1 = select i1 %anot1_, i1 %bnot1_, i1 false
; CHECK: br i1 %c1_unwrap, label %invertbdef, label %staging
; CHECK: staging:
; CHECK-NEXT: br i1 %c2_unwrap, label %invertb1, label %invertb2
