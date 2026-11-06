; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

declare void @_RNvCsbpInmopFggX_7___rustc35___rust_no_alloc_shim_is_unstable_v2()
declare noalias ptr @_RNvCsbpInmopFggX_7___rustc19___rust_alloc_zeroed(i64, i64)

declare noalias ptr @_RNvCs9aBcDeFgHiJ_7___rustc12___rust_alloc(i64, i64)
declare void @_RNvCs9aBcDeFgHiJ_7___rustc14___rust_dealloc(ptr, i64, i64)

declare double @__enzyme_autodiff(...)

define void @allocate_zeroed_box(ptr nocapture %out, double %x) noinline {
entry:
  call void @_RNvCsbpInmopFggX_7___rustc35___rust_no_alloc_shim_is_unstable_v2()
  %buf = call ptr @_RNvCsbpInmopFggX_7___rustc19___rust_alloc_zeroed(i64 16, i64 8)
  store ptr %buf, ptr %out, align 8
  %slot = getelementptr inbounds double, ptr %buf, i64 1
  store double %x, ptr %slot, align 8
  ret void
}

define double @caller_zeroed(double %x) {
entry:
  %box = alloca ptr, align 8
  call void @allocate_zeroed_box(ptr nonnull %box, double %x)
  %ptr = load ptr, ptr %box, align 8
  %slot0 = load double, ptr %ptr, align 8
  %gep1 = getelementptr inbounds double, ptr %ptr, i64 1
  %slot1 = load double, ptr %gep1, align 8
  %sum = fadd double %slot0, %slot1
  %sq = fmul double %sum, %sum
  ret double %sq
}

define double @uninit_box_square(double %x) {
entry:
  call void @_RNvCsbpInmopFggX_7___rustc35___rust_no_alloc_shim_is_unstable_v2()
  %buf = call ptr @_RNvCs9aBcDeFgHiJ_7___rustc12___rust_alloc(i64 8, i64 8)
  store double %x, ptr %buf, align 8
  %val = load double, ptr %buf, align 8
  %sq = fmul double %val, %val
  call void @_RNvCs9aBcDeFgHiJ_7___rustc14___rust_dealloc(ptr %buf, i64 8, i64 8)
  ret double %sq
}

; CHECK-LABEL: define double @diff_zeroed(
define double @diff_zeroed(double %x) {
entry:
  ; CHECK: %[[GRAD0:.+]] = call { double } @diffecaller_zeroed(double %x, double 1.000000e+00)
  ; CHECK: %[[RES0:.+]] = extractvalue { double } %[[GRAD0]], 0
  ; CHECK: ret double %[[RES0]]
  %dx = call double (...) @__enzyme_autodiff(ptr @caller_zeroed, double %x)
  ret double %dx
}

; CHECK-LABEL: define double @diff_uninit(
define double @diff_uninit(double %x) {
entry:
  ; CHECK: %[[GRAD1:.+]] = call { double } @diffeuninit_box_square(double %x, double 1.000000e+00)
  ; CHECK: %[[RES1:.+]] = extractvalue { double } %[[GRAD1]], 0
  ; CHECK: ret double %[[RES1]]
  %dx = call double (...) @__enzyme_autodiff(ptr @uninit_box_square, double %x)
  ret double %dx
}

; CHECK-LABEL: define internal { ptr, ptr } @augmented_allocate_zeroed_box(
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @_RNvCsbpInmopFggX_7___rustc35___rust_no_alloc_shim_is_unstable_v2()
; CHECK-NEXT:   %[[ZSHADOW:.+]] = call noalias nonnull ptr @_RNvCsbpInmopFggX_7___rustc19___rust_alloc_zeroed(i64 16, i64 8)
; CHECK-NEXT:   %[[ZPRIMAL:.+]] = call ptr @_RNvCsbpInmopFggX_7___rustc19___rust_alloc_zeroed(i64 16, i64 8)
; CHECK:        ret { ptr, ptr }

; CHECK-LABEL: define internal { double } @diffeallocate_zeroed_box(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ZTAPE0:.+]] = extractvalue { ptr, ptr } %tapeArg, 0
; CHECK-NEXT:   %[[ZTAPE1:.+]] = extractvalue { ptr, ptr } %tapeArg, 1
; CHECK-NEXT:   %[[ZSLOT:.+]] = getelementptr inbounds double, ptr %[[ZTAPE0]], i64 1
; CHECK-NEXT:   %[[ZDIFF:.+]] = load double, ptr %[[ZSLOT]], align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[ZSLOT]], align 8
; CHECK-NEXT:   call void @_RNvCsbpInmopFggX_7___rustc14___rust_dealloc(ptr nonnull %[[ZTAPE0]], i64 16, i64 8)
; CHECK-NEXT:   call void @_RNvCsbpInmopFggX_7___rustc14___rust_dealloc(ptr %[[ZTAPE1]], i64 16, i64 8)
; CHECK-NEXT:   %[[ZRET:.+]] = insertvalue { double } {{(undef|poison)}}, double %[[ZDIFF]], 0
; CHECK-NEXT:   ret { double } %[[ZRET]]

; CHECK-LABEL: define internal { double } @diffeuninit_box_square(
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @_RNvCsbpInmopFggX_7___rustc35___rust_no_alloc_shim_is_unstable_v2()
; CHECK-NEXT:   %[[USHADOW:.+]] = call noalias nonnull ptr @_RNvCs9aBcDeFgHiJ_7___rustc12___rust_alloc(i64 8, i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull dereferenceable(8) dereferenceable_or_null(8) %[[USHADOW]], i8 0, i64 8, i1 false)
; CHECK-NEXT:   %[[UPRIMAL:.+]] = call ptr @_RNvCs9aBcDeFgHiJ_7___rustc12___rust_alloc(i64 8, i64 8)
; CHECK:        call void @_RNvCs9aBcDeFgHiJ_7___rustc14___rust_dealloc(ptr nonnull %[[USHADOW]], i64 8, i64 8)
; CHECK-NEXT:   call void @_RNvCs9aBcDeFgHiJ_7___rustc14___rust_dealloc(ptr %[[UPRIMAL]], i64 8, i64 8)
; CHECK-NEXT:   %[[URET:.+]] = insertvalue { double } {{(undef|poison)}}, double %{{.+}}, 0
; CHECK-NEXT:   ret { double } %[[URET]]
