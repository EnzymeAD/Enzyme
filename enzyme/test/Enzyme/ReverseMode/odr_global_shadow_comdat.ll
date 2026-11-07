; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.CacheSizes = type { i64, i64, i64 }

$_ZZ20manage_caching_sizesiPlS_E12m_cacheSizes = comdat any
$_ZGVZ20manage_caching_sizesiPlS_E12m_cacheSizes = comdat any
$_ZZ12inline_scaleEdE5state = comdat any
$_ZN6HolderIdE5valueE = comdat any

@_ZZ20manage_caching_sizesiPlS_E12m_cacheSizes = linkonce_odr dso_local global %struct.CacheSizes zeroinitializer, comdat, align 8
@_ZGVZ20manage_caching_sizesiPlS_E12m_cacheSizes = linkonce_odr dso_local global i64 0, comdat, align 8
@_ZZ12inline_scaleEdE5state = linkonce_odr dso_local global double 0.000000e+00, comdat, align 8
@_ZN6HolderIdE5valueE = weak_odr hidden global double 0.000000e+00, comdat, align 8
@global_ptr_table = internal constant [1 x ptr] [ptr @_ZN6HolderIdE5valueE], align 8

; CHECK-DAG: $_ZZ12inline_scaleEdE5state_shadow = comdat any
; CHECK-DAG: $_ZN6HolderIdE5valueE_shadow = comdat any
; CHECK-NOT: _ZGVZ20manage_caching_sizesiPlS_E12m_cacheSizes_shadow
; CHECK: @_ZZ12inline_scaleEdE5state_shadow = linkonce_odr dso_local global double 0.000000e+00, comdat, align 8
; CHECK: @_ZN6HolderIdE5valueE_shadow = weak_odr hidden global double 0.000000e+00, comdat, align 8
; CHECK: @global_ptr_table_shadow = internal constant [1 x ptr] [ptr @_ZN6HolderIdE5valueE_shadow], align 8

declare i32 @__cxa_guard_acquire(ptr)
declare void @__cxa_guard_release(ptr)
declare void @__cxa_guard_abort(ptr)
declare double @__enzyme_autodiff(...)

define linkonce_odr dso_local void @_Z20manage_caching_sizesiPlS_(i32 %action, ptr %l1, ptr %l2) noinline {
entry:
  %guard_byte = load atomic i8, ptr @_ZGVZ20manage_caching_sizesiPlS_E12m_cacheSizes acquire, align 8
  %uninit = icmp eq i8 %guard_byte, 0
  br i1 %uninit, label %init_check, label %init_end

init_check:
  %acquired = call i32 @__cxa_guard_acquire(ptr @_ZGVZ20manage_caching_sizesiPlS_E12m_cacheSizes)
  %tobool = icmp ne i32 %acquired, 0
  br i1 %tobool, label %init_run, label %init_end

init_run:
  store i64 32768, ptr @_ZZ20manage_caching_sizesiPlS_E12m_cacheSizes, align 8
  %f1 = getelementptr inbounds %struct.CacheSizes, ptr @_ZZ20manage_caching_sizesiPlS_E12m_cacheSizes, i64 0, i32 1
  store i64 262144, ptr %f1, align 8
  call void @__cxa_guard_release(ptr @_ZGVZ20manage_caching_sizesiPlS_E12m_cacheSizes)
  br label %init_end

init_end:
  %is_get = icmp eq i32 %action, 0
  br i1 %is_get, label %do_get, label %exit

do_get:
  %v0 = load i64, ptr @_ZZ20manage_caching_sizesiPlS_E12m_cacheSizes, align 8
  store i64 %v0, ptr %l1, align 8
  %p1 = getelementptr inbounds %struct.CacheSizes, ptr @_ZZ20manage_caching_sizesiPlS_E12m_cacheSizes, i64 0, i32 1
  %v1 = load i64, ptr %p1, align 8
  store i64 %v1, ptr %l2, align 8
  br label %exit

exit:
  ret void
}

define internal void @store_via_table(double %x) noinline {
entry:
  store double %x, ptr @_ZZ12inline_scaleEdE5state, align 8
  %slot = load ptr, ptr @global_ptr_table, align 8
  %sq = fmul double %x, %x
  store double %sq, ptr %slot, align 8
  ret void
}

define double @caller(double %x) {
entry:
  %l1 = alloca i64, align 8
  %l2 = alloca i64, align 8
  call void @_Z20manage_caching_sizesiPlS_(i32 0, ptr %l1, ptr %l2)
  call void @store_via_table(double %x)
  %s = load double, ptr @_ZZ12inline_scaleEdE5state, align 8
  %val = load double, ptr @_ZN6HolderIdE5valueE, align 8
  %scaled = fmul double %val, 3.000000e+00
  %res = fadd double %scaled, %s
  ret double %res
}

define double @test_grad(double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @caller, double %x)
  ret double %dx
}

; CHECK-LABEL: define internal { double } @diffecaller(double %x, double %differeturn)
; CHECK: %[[OLD_S:.+]] = load double, ptr @_ZZ12inline_scaleEdE5state_shadow, align 8
; CHECK: %[[SUM_S:.+]] = fadd fast double %[[OLD_S]], %differeturn
; CHECK: store double %[[SUM_S]], ptr @_ZZ12inline_scaleEdE5state_shadow, align 8
; CHECK: call { double } @diffestore_via_table(double %x)
