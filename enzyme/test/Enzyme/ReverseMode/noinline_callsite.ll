; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s

; A call to __enzyme_autodiff marked noinline at the call site, inside a
; function with debug info but without its own !dbg location, is legal IR only
; because it cannot be inlined. The lowered call to the derivative must keep
; the call-site noinline attribute, or the verifier rejects the module with
; "inlinable function call in a function with debug info must have a !dbg
; location".

declare double @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)
declare [2 x double] @__enzyme_batch(...)

define double @square(double %x) {
entry:
  %m = fmul double %x, %x
  ret double %m
}

; CHECK-LABEL: define double @caller(
define double @caller(double %x) !dbg !4 {
entry:
  ; CHECK: %[[rev:.+]] = call { double } @diffesquare(double %x, double 1.000000e+00) #[[REVATTRS:[0-9]+]]
  %d = call double (...) @__enzyme_autodiff(ptr @square, double %x) #0
  ; CHECK: %[[fwd:.+]] = call fast double @fwddiffesquare(double %x, double 1.000000e+00) #[[REVATTRS]]
  %f = call double (...) @__enzyme_fwddiff(ptr @square, double %x, double 1.000000e+00) #0
  ; CHECK: %[[batch:.+]] = call [2 x double] @batch_square([2 x double] %{{.+}}) #[[REVATTRS]]
  %b = call [2 x double] (...) @__enzyme_batch(ptr @square, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x, double %x) #0
  %b0 = extractvalue [2 x double] %b, 0
  %s = fadd double %d, %f
  %t = fadd double %s, %b0
  ret double %t
}

; CHECK: attributes #[[REVATTRS]] = { noinline }

attributes #0 = { noinline }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "noinline_callsite.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
