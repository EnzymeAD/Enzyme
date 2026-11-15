; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -enzyme-global-activity=1 -S | FileCheck %s; fi

; Under global activity the shadow of a global (and its enzyme_shadow
; metadata) is created when it is first needed. A load whose pointer is the
; global itself must still add its adjoint into that shadow, rather than be
; taken for a load from an inactive, unmarked global. This is MITgcm's
;   CALL THE_MAIN_LOOP( ... )
;   cost = fc
; with fc the first variable of COMMON /COST_R/ and the model state in
; another COMMON block; dropping it lost the whole adjoint seed.

@cost_r_ = global [16 x i8] zeroinitializer, align 8
@state_ = global [8 x i8] zeroinitializer, align 8

define internal void @the_main_loop_() noinline {
  %x = load double, ptr @state_, align 8
  %f = fmul double %x, %x
  store double %f, ptr @cost_r_, align 8
  ret void
}

define void @enzyme_fc_(ptr noalias %cost) {
  tail call void @the_main_loop_()
  %fc = load double, ptr @cost_r_, align 8
  store double %fc, ptr %cost, align 8
  ret void
}

declare void @__enzyme_autodiff(...)

define void @test(ptr %c, ptr %dc) {
  call void (...) @__enzyme_autodiff(ptr @enzyme_fc_, metadata !"enzyme_dup", ptr %c, ptr %dc)
  ret void
}

; CHECK: define internal void @diffeenzyme_fc_(ptr {{.*}}%cost, ptr {{.*}}%"cost'")
; CHECK:   call void @augmented_the_main_loop_()
; CHECK:   %[[dc:.+]] = load double, ptr %"cost'", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"cost'", align 8
; CHECK-NEXT:   %[[old:.+]] = load double, ptr @cost_r__shadow, align 8
; CHECK-NEXT:   %[[new:.+]] = fadd fast double %[[old]], %[[dc]]
; CHECK-NEXT:   store double %[[new]], ptr @cost_r__shadow, align 8
; CHECK-NEXT:   call void @diffethe_main_loop_()

; CHECK: define internal void @diffethe_main_loop_()
; CHECK:   %[[df:.+]] = load double, ptr @cost_r__shadow, align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr @cost_r__shadow, align 8
; CHECK:   %[[sx:.+]] = load double, ptr @state__shadow, align 8
; CHECK-NEXT:   %[[sx2:.+]] = fadd fast double %[[sx]], %{{.+}}
; CHECK-NEXT:   store double %[[sx2]], ptr @state__shadow, align 8
