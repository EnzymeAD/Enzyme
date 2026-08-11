; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S -enzyme-julia-addr-load -enzyme-detect-readthrow=0 | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S -enzyme-julia-addr-load -enzyme-detect-readthrow=0 | FileCheck %s

declare void @__enzyme_reverse(...)

define double @square(double **%x, double *%y, i1 %cond) {
entry:
  %ld0 = load double*, double** %x

  %iv1 = insertvalue [2 x double*] undef, double* %ld0, 0
  %iv2 = insertvalue [2 x double*] %iv1, double* %y, 1

  %ev = extractvalue [2 x double*] %iv2, 0
  %ad = addrspacecast double* %ev to double addrspace(1)*
  %res = load double, double addrspace(1)* %ad
  ret double %res
}

define void @dsquare(double *%xp, double *%dxp, double %x) {
entry:
  call void (...) @__enzyme_reverse(double (double**, double*, i1)* nonnull @square,  metadata !"enzyme_const", double *%xp, double *%xp, double *%dxp, i1 true, double %x, i8* null)
  ret void
}

; CHECK: define internal { i8*, double } @augmented_square(double** %x, double* %y, double* %"y'", i1 %cond)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { i8*, double }, align 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %tapemem = bitcast i8* %malloccall to double**
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, double }, { i8*, double }* %0, i32 0, i32 0
; CHECK-NEXT:   store i8* %malloccall, i8** %1, align 8
; CHECK-NEXT:   %ld0 = load double*, double** %x, align 8
; CHECK-NEXT:   store double* %ld0, double** %tapemem, align 8
; CHECK-NEXT:   %ad = addrspacecast double* %ld0 to double addrspace(1)*
; CHECK-NEXT:   %res = load double, double addrspace(1)* %ad, align 8
; CHECK-NEXT:   %2 = getelementptr inbounds { i8*, double }, { i8*, double }* %0, i32 0, i32 1
; CHECK-NEXT:   store double %res, double* %2, align 8
; CHECK-NEXT:   %3 = load { i8*, double }, { i8*, double }* %0, align 8
; CHECK-NEXT:   ret { i8*, double } %3
; CHECK-NEXT: }

; CHECK: define internal void @diffesquare(double** %x, double* %y, double* %"y'", i1 %cond, double %differeturn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to double**
; CHECK-NEXT:   %ld0 = load double*, double** %0, align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %"ad'ipc" = addrspacecast double* %ld0 to double addrspace(1)*
; CHECK-NEXT:   %1 = load double, double addrspace(1)* %"ad'ipc", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   store double %2, double addrspace(1)* %"ad'ipc", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

