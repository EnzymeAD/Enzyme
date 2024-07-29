; RUN: if [ %llvmver -eq 15 ]; then %opt < %s %loadEnzyme -enzyme -opaque-pointers=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -opaque-pointers=1 -S | FileCheck %s; fi

; ModuleID = 'out.ll'
source_filename = "ad.ab3e8598fedbb3bd-cgu.0"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@anon.a6304942fd3c743a92a7245cba3cd4a6.2 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.a6304942fd3c743a92a7245cba3cd4a6.3 = external hidden unnamed_addr constant <{}>, align 8

; Function Attrs: noinline nonlazybind sanitize_hwaddress uwtable
define dso_local "enzyme_type"="{}" void @eprintfunc() unnamed_addr #0 {
  %1 = alloca { { ptr, i64 }, { ptr, i64 }, { ptr, i64 } }, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %1)
  store ptr @anon.a6304942fd3c743a92a7245cba3cd4a6.2, ptr %1, align 8
  %2 = getelementptr inbounds { ptr, i64 }, ptr %1, i64 0, i32 1
  store i64 1, ptr %2, align 8
  %3 = getelementptr inbounds { { ptr, i64 }, { ptr, i64 }, { ptr, i64 } }, ptr %1, i64 0, i32 2
  store ptr null, ptr %3, align 8
  %4 = getelementptr inbounds { { ptr, i64 }, { ptr, i64 }, { ptr, i64 } }, ptr %1, i64 0, i32 1
  store ptr @anon.a6304942fd3c743a92a7245cba3cd4a6.3, ptr %4, align 8
  %5 = getelementptr inbounds { { ptr, i64 }, { ptr, i64 }, { ptr, i64 } }, ptr %1, i64 0, i32 1, i32 1
  store i64 0, ptr %5, align 8
  call void @_ZN3std2io5stdio7_eprint17h557bd237e67376ddE(ptr noalias nocapture noundef nonnull align 8 dereferenceable(48) %1)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %1)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nonlazybind sanitize_hwaddress uwtable
declare hidden void @_ZN3std2io5stdio7_eprint17h557bd237e67376ddE(ptr noalias nocapture noundef readonly align 8 dereferenceable(48)) unnamed_addr #2

declare void @__enzyme_autodiff(...)

define void @enzyme_opt_helper_0() {
  call void (...) @__enzyme_autodiff(ptr @eprintfunc)
  ret void
}

attributes #0 = { noinline nonlazybind sanitize_hwaddress uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { nonlazybind sanitize_hwaddress uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
!2 = !{i32 2, !"RtLibUseGOT", i32 1}
!3 = !{i32 1, !"LTOPostLink", i32 1}
!4 = !{!"rustc version 1.77.0-nightly (e3875bc6d 2024-07-26)"}

; CHECK: define internal void @diffeeprintfunc() 
; CHECK:  call void @_ZN3std2io5stdio7_eprint17h557bd237e67376ddE(ptr noalias nocapture noundef nonnull align 8 dereferenceable(48) %1) #4
; CHECK:  br label %invert

; CHECK: invert:                                           ; preds = %0
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
