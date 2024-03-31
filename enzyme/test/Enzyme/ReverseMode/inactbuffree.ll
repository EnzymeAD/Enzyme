; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

define noalias double* @alloc() nofree {
entry:
  %call = tail call dereferenceable_or_null(8) i8* @malloc(i64 8)
  %0 = bitcast i8* %call to double*
  ret double* %0
}

declare noalias i8* @malloc(i64) 

define void @dealloc(double* nocapture readonly %ptr) argmemonly {
entry:
  %0 = bitcast double* %ptr to i8*
  tail call void @free(i8* %0)
  ret void
}

declare void @free(i8* nocapture) 

declare double @fib() readnone

define double @square(double %x, i64 %i) {
entry:
  %call = tail call double* @alloc()
  store double 3.0, double* %call, align 8
  %arrayidx2 = getelementptr inbounds double, double* %call, i64 %i
  %ld = load double, double* %arrayidx2, align 8
  %mul = fmul double %ld, %x
  tail call void @dealloc(double* nonnull %call)
  ret double %mul
}

define double @dsquare(double %x) {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double, i64)* @square to i8*), double %x, i64 0)
  ret double %call
}

declare double @__enzyme_autodiff(i8*, double, i64)

; CHECK: define internal { double } @diffesquare(double %x, i64 %i, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call double* @alloc() #3
; CHECK-NEXT:   store double 3.000000e+00, double* %call
; CHECK-NEXT:   %arrayidx2 = getelementptr inbounds double, double* %call, i64 %i
; CHECK-NEXT:   %ld = load double, double* %arrayidx2
; CHECK-NEXT:   tail call void @dealloc(double* nonnull %call)
; CHECK-NEXT:   %0 = fmul fast double %differeturn, %ld
; CHECK-NEXT:   %1 = insertvalue { double } {{(undef|poison)}}, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }
