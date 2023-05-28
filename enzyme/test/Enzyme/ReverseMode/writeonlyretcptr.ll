; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -ge 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -enzyme-zero-cache=0 | FileCheck -check-prefixes CHECK,UNDEF %s; fi
; RUN: if [ %llvmver -ge 11 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-zero-cache=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck -check-prefixes CHECK,UNDEF %s; fi
; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -ge 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -enzyme-zero-cache=1 | FileCheck -check-prefixes CHECK,ZERO %s; fi
; RUN: if [ %llvmver -ge 11 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-zero-cache=1 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck -check-prefixes CHECK,ZERO %s; fi

declare dso_local double @__enzyme_reverse(...)

define void @subsq(double ** writeonly nocapture nonnull noundef %out, double *%r) {
entry:
  store double* %r, double** %out, align 8
  ret void
}

define double @mid(double** %rp, double %x) {
  %r = alloca double, align 8
  store double %x, double* %r
  call void @subsq(double** nonnull noundef %rp, double * %r)
  %ld1 = load double*, double** %rp, align 8
  %ld = load double, double* %ld1, align 8
  ret double %ld
}

define double @dsquare(double %x) local_unnamed_addr {
entry:
  %call = tail call double (...) @__enzyme_reverse(i8* bitcast (double (double**, double)* @mid to i8*), metadata !"enzyme_const", double** null, double %x, double 1.0, i8* null)
  ret double %call
}

; CHECK: define internal { double } @diffemid(double** %rp, double %x, double %differeturn, i8* %tapeArg) 
; THIS MUST NOT CONTAIN NOUNDEF OR NONNULL
; UNDEF:  call void @diffesubsq(double** undef, double* %r, double* %"r'ipc")
; ZERO:  call void @diffesubsq(double** null, double* %r, double* %"r'ipc")

; THIS MUST NOT CONTAIN NOUNDEF OR NONNULL
; CHECK: define internal void @diffesubsq(double** nocapture writeonly %out, double* %r, double* %"r'") 
