; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; The augmented forward pass of a custom derivative may read or write the
; shadow of its arguments. The shadow of a float-only allocation must therefore
; exist in the forward pass when the allocation is passed to a custom
; derivative, both directly (@direct) and through a function which Enzyme
; differentiates itself (@indirect), including through a select (@selected).

declare noalias i8* @malloc(i64)
declare void @free(i8*)

define internal i8* @augment_scale(double* %r, double* %dr) {
entry:
  store double 0.000000e+00, double* %dr, align 8
  %v = load double, double* %r, align 8
  %m = fmul fast double %v, 2.000000e+00
  store double %m, double* %r, align 8
  ret i8* null
}

define internal void @gradient_scale(double* %r, double* %dr, i8* %tapeArg) {
entry:
  %v = load double, double* %dr, align 8
  %m = fmul fast double %v, 2.000000e+00
  store double %m, double* %dr, align 8
  ret void
}

declare !enzyme_augment !{i8* (double*, double*)* @augment_scale} !enzyme_gradient !{void (double*, double*, i8*)* @gradient_scale} void @scale(double* nocapture %r)

define internal void @helper(double* nocapture %r) noinline {
entry:
  call void @scale(double* %r)
  ret void
}

define internal void @helper_sel(double* nocapture %r, i1 %c) noinline {
entry:
  %r1 = getelementptr inbounds double, double* %r, i64 1
  %s = select i1 %c, double* %r, double* %r1
  call void @scale(double* %s)
  ret void
}

define double @direct(double %x) {
entry:
  %p = call noalias i8* @malloc(i64 8)
  %r = bitcast i8* %p to double*
  store double %x, double* %r, align 8
  call void @scale(double* %r)
  %res = load double, double* %r, align 8
  call void @free(i8* %p)
  ret double %res
}

define double @indirect(double %x) {
entry:
  %p = call noalias i8* @malloc(i64 8)
  %r = bitcast i8* %p to double*
  store double %x, double* %r, align 8
  call void @helper(double* %r)
  %res = load double, double* %r, align 8
  call void @free(i8* %p)
  ret double %res
}

define double @selected(double %x, i1 %c) {
entry:
  %p = call noalias i8* @malloc(i64 16)
  %r = bitcast i8* %p to double*
  store double %x, double* %r, align 8
  call void @helper_sel(double* %r, i1 %c)
  %res = load double, double* %r, align 8
  call void @free(i8* %p)
  ret double %res
}

declare { i8*, double } @__enzyme_augmentfwd(...)

define { i8*, double } @test_direct(double %x) {
entry:
  %0 = call { i8*, double } (...) @__enzyme_augmentfwd(double (double)* @direct, double %x)
  ret { i8*, double } %0
}

define { i8*, double } @test_indirect(double %x) {
entry:
  %0 = call { i8*, double } (...) @__enzyme_augmentfwd(double (double)* @indirect, double %x)
  ret { i8*, double } %0
}

define { i8*, double } @test_selected(double %x, i1 %c) {
entry:
  %0 = call { i8*, double } (...) @__enzyme_augmentfwd(double (double, i1)* @selected, double %x, i1 %c)
  ret { i8*, double } %0
}

; CHECK: define internal { i8*, double } @augmented_direct(double %x)
; CHECK: %"p'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK: call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"p'mi", i8 0, i64 8, i1 false)
; CHECK: call i8* @augment_scale(double* %r, double* %"r'ipc")

; CHECK: define internal i8* @augmented_helper(double* nocapture %r, double* nocapture %"r'")
; CHECK: call i8* @augment_scale(double* %r, double* %"r'")

; CHECK: define internal { i8*, double } @augmented_indirect(double %x)
; CHECK: %"p'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK: call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"p'mi", i8 0, i64 8, i1 false)
; CHECK: call i8* @augmented_helper(double* %r, double* %"r'ipc")

; CHECK: define internal i8* @augmented_helper_sel(double* nocapture %r, double* nocapture %"r'", i1 %c)
; CHECK: %"s'ipse" = select i1 %c, double* %"r'", double* %"r1'ipg"
; CHECK: call i8* @augment_scale(double* %s, double* %"s'ipse")

; CHECK: define internal { i8*, double } @augmented_selected(double %x, i1 %c)
; CHECK: %"p'mi" = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK: store i8* %"p'mi", i8** %{{.+}}
; CHECK: call i8* @augmented_helper_sel(double* %r, double* %"r'ipc", i1 %c)
