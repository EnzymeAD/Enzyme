; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

declare noalias i8* @_Znwm(i64)

define void @caller() {
entry:
  %raw = call noalias i8* @_Znwm(i64 24)
  %v = bitcast i8* %raw to [3 x double]*
  %v0 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 0
  store double 0.000000, double* %v0, align 8, !tbaa !2
  %v1 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 1
  store double 0.000000, double* %v1, align 8, !tbaa !2
  %v2 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 2
  store double 0.000000, double* %v2, align 8, !tbaa !2
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"double"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %raw = call noalias {{(i8\*|ptr)}} @_Znwm(i64 24): {[-1]:Pointer, [-1,-1]:Float@double}
