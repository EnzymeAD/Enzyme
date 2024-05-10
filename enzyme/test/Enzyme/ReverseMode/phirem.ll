; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-julia-addr-load -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false  -enzyme-julia-addr-load -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify)" -S | FileCheck %s

declare double @__enzyme_reverse(...)

declare {}*** @julia.get_pgcstack() nofree readnone

declare void @julia.safepoint(i64*) nofree

declare i8* @malloc(i64)

define void @test_derivative() {
entry:
  %r = tail call double (...) @__enzyme_reverse(double (double, i64)* nonnull @f, double 1.0, i64 1, double 1.0, i8* null)
  ret void
}

define double @f(double %x, i64 %v) {
entry:
  %a2 = alloca [1 x double]*
  %bc = bitcast [1 x double]** %a2 to double**
  %val = add i64 %v, 1
  call fastcc void @d(double** noalias nocapture nofree writeonly %bc, i64 %val)
  %ptr = load double*, double** %bc
  %cst = load double, double* %ptr
  %mul = fmul double %cst, %x
  ret double %mul
}

define internal fastcc void @d(double** noalias nocapture writeonly "enzyme_inactive" %a0, i64 "enzyme_inactive" %a1) {
pass14:
  %a7 = call i8* @malloc(i64 8)
  %bc = bitcast i8* %a7 to double*
  %fp = sitofp i64 %a1 to double
  store double %fp, double* %bc
  store double* %bc, double** %a0
  ret void
}

; CHECK: define internal fastcc i8* @augmented_d(double** noalias nocapture writeonly "enzyme_inactive" %a0, i64 "enzyme_inactive" %a1)
; CHECK-NEXT: pass14:
; CHECK-NEXT:   %a7 = call i8* @malloc(i64 8)
; CHECK-NEXT:   %bc = bitcast i8* %a7 to double*
; CHECK-NEXT:   %fp = sitofp i64 %a1 to double
; CHECK-NEXT:   store double %fp, double* %bc, align 8
; CHECK-NEXT:   store double* %bc, double** %a0, align 8
; CHECK-NEXT:   ret i8* %a7
; CHECK-NEXT: }

; CHECK: define internal { i8*, double } @augmented_f(double %x, i64 %v)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { i8*, double }, align 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   %tapemem = bitcast i8* %malloccall to { i8*, double }*
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, double }, { i8*, double }* %0, i32 0, i32 0
; CHECK-NEXT:   store i8* %malloccall, i8** %1, align 8
; CHECK-NEXT:   %a2 = alloca [1 x double]*, i64 1, align 8
; CHECK-NEXT:   %bc = bitcast [1 x double]** %a2 to double**
; CHECK-NEXT:   %val = add i64 %v, 1
; CHECK-NEXT:   %_augmented = call fastcc i8* @augmented_d(double** nocapture nofree writeonly %bc, i64 %val)
; CHECK-NEXT:   %2 = getelementptr inbounds { i8*, double }, { i8*, double }* %tapemem, i32 0, i32 0
; CHECK-NEXT:   store i8* %_augmented, i8** %2, align 8
; CHECK-NEXT:   %ptr = load double*, double** %bc, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %cst = load double, double* %ptr, align 8, !alias.scope !5, !noalias !8
; CHECK-NEXT:   %3 = getelementptr inbounds { i8*, double }, { i8*, double }* %tapemem, i32 0, i32 1
; CHECK-NEXT:   store double %cst, double* %3, align 8
; CHECK-NEXT:   %mul = fmul double %cst, %x
; CHECK-NEXT:   %4 = getelementptr inbounds { i8*, double }, { i8*, double }* %0, i32 0, i32 1
; CHECK-NEXT:   store double %mul, double* %4, align 8
; CHECK-NEXT:   %5 = load { i8*, double }, { i8*, double }* %0, align 8
; CHECK-NEXT:   ret { i8*, double } %5
; CHECK-NEXT: }

; CHECK: define internal { double } @diffef(double %x, i64 %v, double %differeturn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { i8*, double }*
; CHECK-NEXT:   %truetape = load { i8*, double }, { i8*, double }* %0, align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %val = add i64 %v, 1
; CHECK-NEXT:   %tapeArg1 = extractvalue { i8*, double } %truetape, 0
; CHECK-NEXT:   %cst = extractvalue { i8*, double } %truetape, 1
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %cst
; CHECK-NEXT:   call fastcc void @diffed(double** nocapture nofree writeonly undef, i64 %val, i8* %tapeArg1)
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }

; CHECK: define internal fastcc void @diffed(double** noalias nocapture writeonly "enzyme_inactive" %a0, i64 "enzyme_inactive" %a1, i8* %a7)
; CHECK-NEXT: pass14:
; CHECK-NEXT:   call void @free(i8* %a7)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }