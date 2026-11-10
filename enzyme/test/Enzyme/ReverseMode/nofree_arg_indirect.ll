; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

define void @free_evaluate(i8* %model, double* %in, double* %out) {
entry:
  %vtable_ptr = bitcast i8* %model to void (i8*, double*)***
  %vtable = load void (i8*, double*)**, void (i8*, double*)*** %vtable_ptr, align 8
  %vfn_slot = getelementptr inbounds void (i8*, double*)*, void (i8*, double*)** %vtable, i64 2
  %vfn = load void (i8*, double*)*, void (i8*, double*)** %vfn_slot, align 8
  call void %vfn(i8* %model, double* %in)
  %val = load double, double* %in, align 8
  %sq = fmul double %val, %val
  store double %sq, double* %out, align 8
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

define void @caller(i8* %model, double* %in, double* %din, double* %out, double* %dout) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i8*, double*, double*)* @free_evaluate to i8*), metadata !"enzyme_const", i8* %model, metadata !"enzyme_dup", double* %in, double* %din, metadata !"enzyme_dup", double* %out, double* %dout)
  ret void
}

; CHECK: define internal void @diffefree_evaluate({{i8\*|ptr}} %model, {{double\*|ptr}} %in, {{double\*|ptr}} %"in'", {{double\*|ptr}} nocapture writeonly %out, {{double\*|ptr}} nocapture %"out'")
; CHECK: call void %vfn({{i8\*|ptr}} %model, {{double\*|ptr}} %in)
; CHECK: fadd fast double
