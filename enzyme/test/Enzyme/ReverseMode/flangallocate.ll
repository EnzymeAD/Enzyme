; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-globals-default-inactive -enzyme-global-activity -S | FileCheck %s; fi

; A work array allocated and deallocated through the LLVM flang runtime, as
;   allocate(work(n)); work(1) = x; y = work(1) * work(1); deallocate(work)
; for a module variable `work` with a declared shadow. The augmented pass
; replays the allocation on the shadow descriptor and zeroes the new shadow
; memory; the deallocation only detaches the shadow memory, since the reverse
; pass accumulates into it. The reverse pass attaches it again and finally
; deallocates it through the runtime. Calls on inactive descriptors stay
; primal only.

@work = dso_local global { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { ptr null, i64 8, i32 20240719, i8 1, i8 28, i8 2, i8 0, [1 x [3 x i64]] zeroinitializer }, align 8, !enzyme_shadow !4
@work_shadow = dso_local global { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } { ptr null, i64 8, i32 20240719, i8 1, i8 28, i8 2, i8 0, [1 x [3 x i64]] zeroinitializer }, align 8
@clock = dso_local global { ptr, i64, i32, i8, i8, i8, i8 } { ptr null, i64 8, i32 20240719, i8 0, i8 10, i8 2, i8 0 }, align 8
@clock_new = dso_local global { ptr, i64, i32, i8, i8, i8, i8 } { ptr null, i64 8, i32 20240719, i8 0, i8 10, i8 2, i8 0 }, align 8

declare void @_FortranAAllocatableSetBounds(ptr, i32, i64, i64)
declare i32 @_FortranAAllocatableAllocate(ptr, ptr, i1, ptr, ptr, i32, ptr)
declare i32 @_FortranAAllocatableDeallocate(ptr, i1, ptr, ptr, i32)
declare void @_FortranAAssignTemporary(ptr, ptr, ptr, i32)
declare ptr @_FortranAMemcpyWrapper(ptr, ptr, i64)

define double @square(i64 %n, double %x) {
entry:
  call void @_FortranAAssignTemporary(ptr @clock, ptr @clock_new, ptr null, i32 0)
  call void @_FortranAAllocatableSetBounds(ptr @work, i32 0, i64 1, i64 %n)
  %st = call i32 @_FortranAAllocatableAllocate(ptr @work, ptr null, i1 false, ptr null, ptr null, i32 0, ptr @_FortranAMemcpyWrapper)
  %base = load ptr, ptr @work, align 8, !tbaa !0
  store double %x, ptr %base, align 8, !tbaa !5
  %v = load double, ptr %base, align 8, !tbaa !5
  %y = fmul double %v, %v
  %st2 = call i32 @_FortranAAllocatableDeallocate(ptr @work, i1 false, ptr null, ptr null, i32 0)
  ret double %y
}

define double @dsquare(i64 %n, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @square, metadata !"enzyme_const", i64 %n, double %x)
  ret double %r
}

declare double @__enzyme_autodiff(...)

define void @dsquare_split(i64 %n, double %x) {
entry:
  %aug = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @square, metadata !"enzyme_const", i64 %n, double %x)
  %tape = extractvalue { ptr, double } %aug, 0
  %r = call double (...) @__enzyme_reverse(ptr @square, metadata !"enzyme_const", i64 %n, double %x, double 1.0, ptr %tape)
  ret void
}

declare { ptr, double } @__enzyme_augmentfwd(...)
declare double @__enzyme_reverse(...)

!0 = !{!1, !1, i64 0}
!1 = !{!"descriptor member", !2, i64 0}
!2 = !{!"any access", !3, i64 0}
!3 = !{!"Flang function root square"}
!4 = !{ptr @work_shadow}
!5 = !{!6, !6, i64 0}
!6 = !{!"allocated data/work", !7, i64 0}
!7 = !{!"allocated data", !8, i64 0}
!8 = !{!"target data", !2, i64 0}

; CHECK: define internal { double } @diffesquare(i64 %n, double %x, double %differeturn)
; CHECK:        call void @_FortranAAssignTemporary(ptr @clock, ptr @clock_new, ptr null, i32 0)
; CHECK-NEXT:   call void @_FortranAAllocatableSetBounds(ptr @work, i32 0, i64 1, i64 %n)
; CHECK-NEXT:   call void @_FortranAAllocatableSetBounds(ptr @work_shadow, i32 0, i64 1, i64 %n)
; CHECK-NEXT:   %st = call i32 @_FortranAAllocatableAllocate(ptr @work, ptr null, i1 false, ptr null, ptr null, i32 0, ptr @_FortranAMemcpyWrapper)
; CHECK-NEXT:   {{%.+}} = call i32 @_FortranAAllocatableAllocate(ptr @work_shadow, ptr null, i1 false, ptr null, ptr null, i32 0, ptr @_FortranAMemcpyWrapper)
; CHECK-NEXT:   [[SIZE:%.+]] = call i64 @_FortranASize(ptr @work_shadow, ptr null, i32 0)
; CHECK-NEXT:   [[LEN:%.+]] = load i64, ptr getelementptr inbounds (i8, ptr @work_shadow, i64 8)
; CHECK-NEXT:   [[ABASE:%.+]] = load ptr, ptr @work_shadow
; CHECK-NEXT:   [[BYTES:%.+]] = mul i64 [[SIZE]], [[LEN]]
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr {{.*}}[[ABASE]], i8 0, i64 [[BYTES]], i1 false)
; CHECK-NEXT:   call void @_FortranAInitialize(ptr @work_shadow, ptr null, i32 0)
; CHECK:        %st2 = call i32 @_FortranAAllocatableDeallocate(ptr @work, i1 false, ptr null, ptr null, i32 0)
; CHECK-NEXT:   [[DBASE:%.+]] = load ptr, ptr @work_shadow
; CHECK-NEXT:   store ptr null, ptr @work_shadow
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK:        store ptr [[DBASE]], ptr @work_shadow
; CHECK:        fadd fast double
; CHECK:        store double {{.*}}, ptr %"base'ipl"
; CHECK:        store ptr [[ABASE]], ptr @work_shadow
; CHECK-NEXT:   call i32 @_FortranAAllocatableDeallocate(ptr @work_shadow, i1 true, ptr null, ptr null, i32 0)
; CHECK:        ret { double }

; In split mode the augmented pass tapes both shadow base addresses, and the
; reverse pass makes no primal runtime call.

; CHECK: define internal { ptr, double } @augmented_square(i64 %n, double %x)
; CHECK:        call i32 @_FortranAAllocatableAllocate(ptr @work_shadow,
; CHECK:        [[AB:%.+]] = load ptr, ptr @work_shadow
; CHECK:        insertvalue [1 x ptr] undef, ptr [[AB]], 0
; CHECK:        call i32 @_FortranAAllocatableDeallocate(ptr @work, i1 false, ptr null, ptr null, i32 0)
; CHECK-NEXT:   [[DB:%.+]] = load ptr, ptr @work_shadow
; CHECK-NEXT:   store ptr null, ptr @work_shadow
; CHECK-NEXT:   insertvalue [1 x ptr] undef, ptr [[DB]], 0
; CHECK:        ret { ptr, double }

; CHECK: define internal { double } @diffesquare.{{[0-9]+}}(i64 %n, double %x, double %differeturn, ptr %tapeArg)
; CHECK-NOT:    call {{.*}}@_FortranAAssignTemporary
; CHECK-NOT:    call {{.*}}(ptr @work,
; CHECK:        store ptr {{%.+}}, ptr @work_shadow
; CHECK:        store ptr {{%.+}}, ptr @work_shadow
; CHECK-NEXT:   call i32 @_FortranAAllocatableDeallocate(ptr @work_shadow, i1 true, ptr null, ptr null, i32 0)
; CHECK:        ret { double }
