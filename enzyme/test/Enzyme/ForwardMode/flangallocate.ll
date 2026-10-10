; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Allocation of a Fortran allocatable through the LLVM flang runtime, as
;   allocate(work(n)); work = x
; for a module variable `work` with a declared shadow. The derivative
; replays the runtime calls on the shadow descriptor, zeroes the newly
; allocated shadow memory and (re)initializes it.

@work = dso_local global { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { ptr null, i64 8, i32 20240719, i8 1, i8 28, i8 2, i8 0, [1 x [3 x i64]] zeroinitializer }, align 8, !enzyme_shadow !4
@work_shadow = dso_local global { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { ptr null, i64 8, i32 20240719, i8 1, i8 28, i8 2, i8 0, [1 x [3 x i64]] zeroinitializer }, align 8

declare void @_FortranAAllocatableSetBounds(ptr, i32, i64, i64)
declare i32 @_FortranAAllocatableAllocate(ptr, ptr, i1, ptr, ptr, i32, ptr)
declare ptr @_FortranAMemcpyWrapper(ptr, ptr, i64)

define void @alloc(i64 %n, double %x) {
entry:
  call void @_FortranAAllocatableSetBounds(ptr @work, i32 0, i64 1, i64 %n)
  %st = call i32 @_FortranAAllocatableAllocate(ptr @work, ptr null, i1 false, ptr null, ptr null, i32 0, ptr @_FortranAMemcpyWrapper)
  %base = load ptr, ptr @work, align 8, !tbaa !0
  store double %x, ptr %base, align 8
  ret void
}

define void @dalloc(i64 %n, double %x, double %dx) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @alloc, metadata !"enzyme_const", i64 %n, double %x, double %dx)
  ret void
}

declare void @__enzyme_fwddiff(...)

!0 = !{!1, !1, i64 0}
!1 = !{!"descriptor member", !2, i64 0}
!2 = !{!"any access", !3, i64 0}
!3 = !{!"Flang function root alloc"}
!4 = !{ptr @work_shadow}

; CHECK: define internal void @fwddiffealloc(i64 %n, double %x, double %"x'")
; CHECK:   call void @_FortranAAllocatableSetBounds(ptr @work, i32 0, i64 1, i64 %n)
; CHECK-NEXT:   call void @_FortranAAllocatableSetBounds(ptr @work_shadow, i32 0, i64 1, i64 %n)
; CHECK:   call i32 @_FortranAAllocatableAllocate(ptr @work, ptr null, i1 false, ptr null, ptr null, i32 0, ptr @_FortranAMemcpyWrapper)
; CHECK-NEXT:   call i32 @_FortranAAllocatableAllocate(ptr @work_shadow, ptr null, i1 false, ptr null, ptr null, i32 0, ptr @_FortranAMemcpyWrapper)
; CHECK-NEXT:   [[SIZE:%.+]] = call i64 @_FortranASize(ptr @work_shadow, ptr null, i32 0)
; CHECK-NEXT:   [[LEN:%.+]] = load i64, ptr getelementptr inbounds (i8, ptr @work_shadow, i64 8)
; CHECK-NEXT:   [[DBASE:%.+]] = load ptr, ptr @work_shadow
; CHECK-NEXT:   [[BYTES:%.+]] = mul i64 [[SIZE]], [[LEN]]
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr {{.*}}[[DBASE]], i8 0, i64 [[BYTES]], i1 false)
; CHECK-NEXT:   call void @_FortranAInitialize(ptr @work_shadow, ptr null, i32 0)
; CHECK:   store double %"x'", ptr
; CHECK:   store double %x, ptr
