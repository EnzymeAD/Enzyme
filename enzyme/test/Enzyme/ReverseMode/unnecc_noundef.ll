; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,early-cse)" -S -opaque-pointers | FileCheck %s

declare void @__enzyme_autodiff(...)

define void @f(ptr noundef nonnull %W, ptr noundef nonnull %Wp, ptr noundef nonnull %M, ptr noundef nonnull %Mp) {
  call void (...) @__enzyme_autodiff(ptr noundef nonnull @matvec, ptr noundef nonnull %W, ptr noundef nonnull %Wp, ptr noundef nonnull %M, ptr noundef nonnull %Mp)
  ret void
} 

define internal void @matvec(ptr noalias noundef %W, ptr noalias noundef readonly nocapture %output) {
entry:
  %i16 = load ptr, ptr %output, align 8
  call void @mat(ptr noundef nonnull align 8 dereferenceable(16) %i16, ptr noundef nonnull align 8 dereferenceable(16) %W)
  %g = getelementptr i8, ptr %i16, i64 32
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %g, ptr align 8 %W, i64 8, i1 false), !enzyme_truetype !0
  ret void
}

define linkonce_odr dso_local void @mat(ptr nocapture %dst, ptr nocapture readonly noundef nonnull align 8 dereferenceable(16) %rhs) {
entry:
  %ref.tmp8.i = alloca { ptr, i32 }, align 8
  %i15 = load ptr, ptr %rhs, align 8
  store ptr %i15, ptr %ref.tmp8.i, align 8
  %g2 = getelementptr { ptr, i32 }, ptr %ref.tmp8.i, i64 0, i32 1
  store i32 0, ptr %g2, align 8
  %q = call <2 x double> @last(ptr %ref.tmp8.i)
  store <2 x double> %q, ptr %dst, align 1
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

define linkonce_odr dso_local <2 x double> @last(ptr nocapture readonly %rhs) {
entry:
  %i13 = load ptr, ptr %rhs, align 8
  %g3 = getelementptr { ptr, i32 }, ptr %rhs, i64 0, i32 1
  %s = load i32, ptr %g3
  %g4 = getelementptr <2 x double>, ptr %i13, i32 %s
  %m = load <2 x double>, ptr %g4, align 1
  ret <2 x double > %m ; void
}

!0 = !{!"Float@float", i64 0}

; Need to check that we don't pass undef to diffelast since we load it in the reverse pass.

; CHECK: define internal void @diffemat(ptr nocapture writeonly %dst, ptr nocapture %"dst'", ptr nocapture readonly align 8 dereferenceable(16) %rhs, ptr nocapture align 8 %"rhs'", { ptr, ptr } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"malloccall'mi" = alloca i8, i64 16, align 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull dereferenceable(16) dereferenceable_or_null(16) %"malloccall'mi", i8 0, i64 16, i1 false)
; CHECK-NEXT:   %malloccall = extractvalue { ptr, ptr } %tapeArg, 1
; CHECK-NEXT:   %"i15'il_phi" = extractvalue { ptr, ptr } %tapeArg, 0
; CHECK-NEXT:   store ptr %"i15'il_phi", ptr %"malloccall'mi", align 8
; CHECK-NEXT:   %"g2'ipg" = getelementptr { ptr, i32 }, ptr %"malloccall'mi", i64 0, i32 1
; CHECK-NEXT:   store i32 0, ptr %"g2'ipg", align 8
; CHECK-NEXT:   %0 = load <2 x double>, ptr %"dst'", align 1
; CHECK-NEXT:   store <2 x double> zeroinitializer, ptr %"dst'"
; CHECK-NEXT:   call void @diffelast(ptr %malloccall, ptr %"malloccall'mi", <2 x double> %0)
; CHECK-NEXT:   call void @free(ptr %malloccall)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffelast(ptr nocapture readonly %rhs, ptr nocapture %"rhs'", <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"i13'ipl" = load ptr, ptr %"rhs'", align 8
; CHECK-NEXT:   %"g3'ipg" = getelementptr { ptr, i32 }, ptr %"rhs'", i64 0, i32 1
; CHECK-NEXT:   %g3 = getelementptr { ptr, i32 }, ptr %rhs, i64 0, i32 1
; CHECK-NEXT:   %s = load i32, ptr %g3, align 4
; CHECK-NEXT:   %"g4'ipg" = getelementptr <2 x double>, ptr %"i13'ipl", i32 %s
; CHECK-NEXT:   %0 = load <2 x double>, ptr %"g4'ipg", align 1
; CHECK-NEXT:   %1 = fadd fast <2 x double> %0, %differeturn
; CHECK-NEXT:   store <2 x double> %1, ptr %"g4'ipg", align 1
; CHECK-NEXT:   ret void
; CHECK-NEXT: }



