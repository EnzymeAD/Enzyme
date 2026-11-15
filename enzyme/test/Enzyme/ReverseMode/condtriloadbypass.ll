; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=0 -enzyme -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -simplifycfg -adce -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=0  -passes="enzyme,function(mem2reg,early-cse,%simplifycfg,instsimplify,correlated-propagation,%simplifycfg,adce)" -S -enzyme-detect-readthrow=0 | FileCheck %s

; Three targets (%bdef, %b1, %b2) merge at %end. %c1 splits %bdef off, but no
; single branch splits %b1 from %b2: %c2 short-circuits straight to %b1, and
; only otherwise does %c4 (two blocks further down, behind a bounds-check style
; block whose other successor is unreachable) choose between %b1 and %b2. The
; reverse pass therefore must not pick %mid3 as the second split block, since
; it does not dominate %b1: doing so would send the %c2 path to %b2 whenever
; %c4 happened to be false. Instead the taken predecessor is cached.

declare double @__enzyme_autodiff(i8*, ...)

declare void @abort()

; Function Attrs: norecurse nounwind uwtable
define double @alldiv(double* %a, i1 %cmp, i32 %val) {
entry:
  br i1 %cmp, label %mid, label %fin

mid:
  %c1 = icmp eq i32 %val, 13
  br i1 %c1, label %bdef, label %mid2

mid2:
  %c2 = icmp eq i32 %val, 17
  br i1 %c2, label %b1, label %chk

chk:
  %c3 = icmp ult i32 %val, 100
  br i1 %c3, label %mid3, label %oob

oob:
  call void @abort()
  unreachable

mid3:
  %c4 = icmp eq i32 %val, 19
  br i1 %c4, label %b1, label %b2

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

; CHECK: define internal void @diffealldiv({{(double\*|ptr)}} %a, {{(double\*|ptr)}} %"a'", i1 %cmp, i32 %val, double %differeturn)

; CHECK: invertend:
; CHECK:   switch i8 %{{.+}}, label %invert{{b1|b2|bdef}} [
; CHECK-NEXT:     i8 {{[0-9]}}, label %invert{{b1|b2|bdef}}
; CHECK-NEXT:     i8 {{[0-9]}}, label %invert{{b1|b2|bdef}}
; CHECK-NEXT:   ]
; CHECK-NOT: staging
