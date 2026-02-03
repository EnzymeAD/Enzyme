; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s

declare { i8*, double } @__enzyme_augmentfwd(...) 

define double* @c(double* %x) readnone {
  ret double* %x
}

define double* @d(double* %x) readnone {
  ret double* %x
}

define double @use(double %x, i1 %z) {
  ret double %x
}

; differential use analysis crash requires
;   %r2 is constant instruction, but not constant value
;   shadow of r2 is not needed
define double @square(double %x, double* %v) {
entry:
  %r1 = call double* @c(double* %v)
  %r2 = call double* @d(double* %r1)
  %z = icmp eq double* %r2, null
  %q = call double @use(double %x, i1 %z)
  ret double %q
}

define void @dsquare(double %x) {
entry:
  %0 = call { i8*, double } (...) @__enzyme_augmentfwd(double (double, double*)* @square, double %x, metadata !"enzyme_dup", double* null, double* null)
  ret void
}

; CHECK: define internal { i8*, double } @augmented_square(double %x, double* %v, double* %"v'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { i8*, double }, align 8
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, double }, { i8*, double }* %0, i32 0, i32 0
; CHECK-NEXT:   store i8* null, i8** %1, align 8
; CHECK-NEXT:   %r1 = call double* @nofree_c(double* %v)
; CHECK-NEXT:   %r2 = call double* @nofree_d(double* %r1)
; CHECK-NEXT:   %z = icmp eq double* %r2, null
; CHECK-NEXT:   %q = call fast double @augmented_use(double %x, i1 %z)
; CHECK-NEXT:   %2 = getelementptr inbounds { i8*, double }, { i8*, double }* %0, i32 0, i32 1
; CHECK-NEXT:   store double %q, double* %2, align 8
; CHECK-NEXT:   %3 = load { i8*, double }, { i8*, double }* %0, align 8
; CHECK-NEXT:   ret { i8*, double } %3
; CHECK-NEXT: }