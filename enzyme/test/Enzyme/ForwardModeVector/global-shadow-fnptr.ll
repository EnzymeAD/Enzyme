; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Calls through a function pointer held by a global that another global
; points to, at width 2: each lane of the shadow table points at the shadow
; of the function.

%struct.Grad2 = type { [2 x double] }

@sqfn = global ptr @sq, align 8
@table = global ptr @sqfn, align 8

define double @sq(double %x) {
entry:
  %m = fmul double %x, %x
  ret double %m
}

define double @f(double %x) {
entry:
  %t = load ptr, ptr @table, align 8
  %fp = load ptr, ptr %t, align 8
  %r = call double %fp(double %x)
  ret double %r
}

declare [2 x double] @__enzyme_fwddiff(...)

define [2 x double] @test(double %x) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.0)
  ret [2 x double] %r
}

; CHECK: @sqfn.ad.w2 = global [2 x ptr] [ptr @"_enzyme_forward2_sq'", ptr @"_enzyme_forward2_sq'"]
; CHECK: @table.ad.w2 = global [2 x ptr] [ptr @sqfn.ad.w2, ptr getelementptr inbounds ([2 x ptr], ptr @sqfn.ad.w2, i32 0, i32 1)]
; CHECK: @"_enzyme_forward2_sq'" = internal constant ptr @fwddiffe2sq
