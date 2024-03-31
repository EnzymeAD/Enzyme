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

define double @square(double %x, i64 %i, i64 %j) {
entry:
  %call = tail call double* @alloc()
  store double 3.0, double* %call, align 8
  %arrayidx2 = getelementptr inbounds double, double* %call, i64 %i
  %ld1 = load double, double* %arrayidx2, align 8
  %mul = fmul double %ld1, %x
  %arrayidx3 = getelementptr inbounds double, double* %call, i64 %j
  %ld2 = load double, double* %arrayidx3, align 8
  %mul2 = fmul double %ld2, %mul
  tail call void @dealloc(double* nonnull %call)
  ret double %mul2
}

define double @dsquare(double %x) {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double, i64, i64)* @square to i8*), double %x, i64 0, i64 0)
  ret double %call
}

declare double @__enzyme_autodiff(i8*, double, i64, i64)

; CHECK: define internal { double } @diffesquare(double %x, i64 %i, i64 %j, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { i8*, double* } @augmented_alloc()
; CHECK-NEXT:   %subcache = extractvalue { i8*, double* } %call_augmented, 0
; CHECK-NEXT:   %call = extractvalue { i8*, double* } %call_augmented, 1
; CHECK:   tail call void @nofree_dealloc(double* nonnull %call)
; CHECK:   call void @diffealloc(i8* %subcache)

; CHECK: define internal void @nofree_dealloc(double* nocapture readonly %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { i8*, double* } @augmented_alloc() 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { i8*, double* }
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, double* }, { i8*, double* }* %0, i32 0, i32 0
; CHECK-NEXT:   %call = tail call dereferenceable_or_null(8) i8* @malloc(i64 8) 
; CHECK-NEXT:   store i8* %call, i8** %1
; CHECK-NEXT:   %2 = bitcast i8* %call to double*
; CHECK-NEXT:   %3 = getelementptr inbounds { i8*, double* }, { i8*, double* }* %0, i32 0, i32 1
; CHECK-NEXT:   store double* %2, double** %3
; CHECK-NEXT:   %4 = load { i8*, double* }, { i8*, double* }* %0, align 8
; CHECK-NEXT:   ret { i8*, double* } %4
; CHECK-NEXT: }

; CHECK: define internal void @diffealloc(i8* %call)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @free(i8* %call)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
