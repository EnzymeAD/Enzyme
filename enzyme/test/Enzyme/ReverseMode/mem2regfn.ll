; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

define double @_Z3food(double %0) {
  ret double %0
}

define i32 @main()  {
  %a5 = alloca i8*, align 8
  store i8* bitcast (double (double)* @_Z3food to i8*), i8** %a5, align 8
  %a17 = load i8*, i8** %a5, align 8
  %q = call double (...) @__enzyme_autodiff(i8* %a17, metadata !"enzyme_dup", double 1.0, double 1.0)
  ret i32 0
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal void @diffe_Z3food
