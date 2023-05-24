; RUN: if [ %llvmver -ge 13 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

%"QNCA_a0$float*$rank1$" = type { float*, i64, i64, i64, i64, i64, [1 x { i64, i64, i64 }] }

define float @selectfirst(%"QNCA_a0$float*$rank1$"* noalias nocapture readonly dereferenceable(72) "ptrnoalias" %X) local_unnamed_addr {
entry:
  %"X.addr_a0$" = getelementptr inbounds %"QNCA_a0$float*$rank1$", %"QNCA_a0$float*$rank1$"* %X, i64 0, i32 0
  %"X.addr_a0$_fetch.29" = load float*, float** %"X.addr_a0$", align 1, !tbaa !0
  %"X.dim_info$.lower_bound$" = getelementptr inbounds %"QNCA_a0$float*$rank1$", %"QNCA_a0$float*$rank1$"* %X, i64 0, i32 6, i64 0, i32 2
  %"X.dim_info$.lower_bound$[]" = tail call i64* @llvm.intel.subscript.p0i64.i64.i32.p0i64.i32(i8 0, i64 0, i32 24, i64* nonnull elementtype(i64) %"X.dim_info$.lower_bound$", i32 0)
  %"X.dim_info$.lower_bound$[]_fetch.30" = load i64, i64* %"X.dim_info$.lower_bound$[]", align 1, !tbaa !6
  %"X.addr_a0$_fetch.29[]" = tail call float* @llvm.intel.subscript.p0f32.i64.i64.p0f32.i64(i8 0, i64 %"X.dim_info$.lower_bound$[]_fetch.30", i64 4, float* elementtype(float) %"X.addr_a0$_fetch.29", i64 1)
  %"X.addr_a0$_fetch.29[]_fetch.32" = load float, float* %"X.addr_a0$_fetch.29[]", align 1, !tbaa !7
  ret float %"X.addr_a0$_fetch.29[]_fetch.32"
}

declare i64* @llvm.intel.subscript.p0i64.i64.i32.p0i64.i32(i8, i64, i32, i64*, i32)
declare float* @llvm.intel.subscript.p0f32.i64.i64.p0f32.i64(i8, i64, i64, float*, i64)

define void @grad_selectfirst(%"QNCA_a0$float*$rank1$"* noalias readonly dereferenceable(72) "ptrnoalias" %X, %"QNCA_a0$float*$rank1$"* noalias dereferenceable(72) "ptrnoalias" %DX) local_unnamed_addr {
entry:
  tail call void @__enzyme_autodiff(float (...)* bitcast (float (%"QNCA_a0$float*$rank1$"*)* @selectfirst to float (...)*), %"QNCA_a0$float*$rank1$"* nonnull %X, %"QNCA_a0$float*$rank1$"* nonnull %DX)
  ret void
}

declare void @__enzyme_autodiff(float (...)* noalias, %"QNCA_a0$float*$rank1$"* noalias readonly dereferenceable(72) "ptrnoalias", %"QNCA_a0$float*$rank1$"* noalias dereferenceable(72) "ptrnoalias") local_unnamed_addr

!0 = !{!1, !2, i64 0}
!1 = !{!"ifx$descr$3", !2, i64 0, !2, i64 8, !2, i64 16, !2, i64 24, !2, i64 32, !2, i64 40, !2, i64 48, !2, i64 56, !2, i64 64}
!2 = !{!"ifx$descr$field", !3, i64 0}
!3 = !{!"Fortran Dope Vector Symbol", !4, i64 0}
!4 = !{!"Generic Fortran Symbol", !5, i64 0}
!5 = !{!"ifx$root$5$selectfirst"}
!6 = !{!1, !2, i64 64}
!7 = !{!8, !8, i64 0}
!8 = !{!"ifx$unique_sym$10", !9, i64 0}
!9 = !{!"Fortran Data Symbol", !4, i64 0}

; CHECK: define internal void @diffeselectfirst(%"QNCA_a0$float*$rank1$"* noalias nocapture readonly dereferenceable(72) "ptrnoalias" %X, %"QNCA_a0$float*$rank1$"* nocapture %"X'", float %differeturn) local_unnamed_addr #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"X.addr_a0$'ipg" = getelementptr inbounds %"QNCA_a0$float*$rank1$", %"QNCA_a0$float*$rank1$"* %"X'", i64 0, i32 0
; CHECK-NEXT:   %"X.addr_a0$" = getelementptr inbounds %"QNCA_a0$float*$rank1$", %"QNCA_a0$float*$rank1$"* %X, i64 0, i32 0
; CHECK-NEXT:   %"X.addr_a0$_fetch.29'ipl" = load float*, float** %"X.addr_a0$'ipg", align 1, !tbaa !0
; CHECK-NEXT:   %"X.addr_a0$_fetch.29" = load float*, float** %"X.addr_a0$", align 1, !tbaa !0
; CHECK-NEXT:   %"X.dim_info$.lower_bound$" = getelementptr inbounds %"QNCA_a0$float*$rank1$", %"QNCA_a0$float*$rank1$"* %X, i64 0, i32 6, i64 0, i32 2
; CHECK-NEXT:   %"X.dim_info$.lower_bound$[]" = tail call i64* @llvm.intel.subscript.p0i64.i64.i32.p0i64.i32(i8 0, i64 0, i32 24, i64* nonnull elementtype(i64) %"X.dim_info$.lower_bound$", i32 0)
; CHECK-NEXT:   %"X.dim_info$.lower_bound$[]_fetch.30" = load i64, i64* %"X.dim_info$.lower_bound$[]", align 1, !tbaa !6
; CHECK-NEXT:   %0 = call float* @llvm.intel.subscript.p0f32.i64.i64.p0f32.i64(i8 0, i64 %"X.dim_info$.lower_bound$[]_fetch.30", i64 4, float* elementtype(float) %"X.addr_a0$_fetch.29'ipl", i64 1)
; CHECK-NEXT:   %"X.addr_a0$_fetch.29[]" = tail call float* @llvm.intel.subscript.p0f32.i64.i64.p0f32.i64(i8 0, i64 %"X.dim_info$.lower_bound$[]_fetch.30", i64 4, float* elementtype(float) %"X.addr_a0$_fetch.29", i64 1)
; CHECK-NEXT:   %1 = load float, float* %0, align 1, !tbaa !7
; CHECK-NEXT:   %2 = fadd fast float %1, %differeturn
; CHECK-NEXT:   store float %2, float* %0, align 1, !tbaa !7
; CHECK-NEXT:   ret void
; CHECK-NEXT: }