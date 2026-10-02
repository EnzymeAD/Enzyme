! !$enzyme fixed_point in front of a loop: flang evaluates the variables in
! front of the loop and passes them to a marker at the start of its body,
! which the plugin replaces with
!   __enzyme_set_fixed_point(reduction, max_iters, control, [state, bytes]...)
! computing each address and size in bytes in front of the loop, as Enzyme
! wants them: a module variable, the data of an allocatable (its descriptor
! loaded in front of the loop), a COMMON member, a whole COMMON block, and an
! explicit-shape dummy of a run-time size. A missing reduction or max_iters is
! -1, a missing control null.
!
! REQUIRES: flang_directives
! RUN: %fc -fc1 %flangFc1Directives -emit-hlfir %s -o - | FileCheck %s --check-prefix=HLFIR
! RUN: %fc -fc1 %flangFc1Directives -O0 -emit-llvm %s -o - | FileCheck %s
! RUN: %fc -fc1 %flangFc1Directives -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=O2

module fp
  implicit none
  real(8) :: u(5)
  real(8), allocatable :: v(:)
  real(8) :: a(3), b
  common /blk/ a, b
  interface
    logical function step()
    end function step
  end interface
contains
  integer function ctl(cumul, reduction)
    real(8), intent(inout) :: cumul, reduction
    ctl = 0
  end function ctl

  subroutine solve()
    !$enzyme fixed_point(u, v, b) reduction(2.d-8) max_iters(100) control(ctl)
    do while (step())
    end do
  end subroutine solve

  subroutine solve_defaults(w, n)
    integer :: n
    real(8) :: w(n)
    integer :: i
    !DIR$ ENZYME FIXED_POINT(w, /blk/)
    do i = 1, n
      w(1) = w(1) * 0.5d0
    end do
  end subroutine solve_defaults

  subroutine solve_keywords()
    !$enzyme fixed_point(u, reduction=-1.5e-3_8, max_iters=0)
    do while (step())
      u(1) = u(1) + 1
    end do
  end subroutine solve_keywords
end module fp

! flang's marker, with the variables evaluated in front of the loop (the
! allocatable's descriptor loaded there).
! HLFIR-LABEL: func.func @_QMfpPsolve()
! HLFIR:         %[[U:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMfpEu"}
! HLFIR:         %[[VREF:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMfpEv"}
! HLFIR:         %[[B:.*]]:2 = hlfir.declare %{{.*}} storage(%{{.*}}[24]) {uniq_name = "_QMfpEb"}
! HLFIR:         %[[V:.*]] = fir.load %[[VREF]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
! HLFIR:         fir.call @_QPstep()
! HLFIR:         cf.cond_br %{{.*}}, ^[[BODY:bb[0-9]+]], ^
! HLFIR:       ^[[BODY]]:
! HLFIR-NEXT:    fir.call @__flang_directive.enzyme.fixed_point(%[[U]]#0, %[[V]], %[[B]]#0) {{.*}}{fir.directive = {args = {control = @_QMfpPctl, max_iters = 100 : i64, reduction = 2.000000e-08 : f64}, keyword = "fixed_point", prefix = "enzyme"}} : (!fir.ref<!fir.array<5xf64>>, !fir.box<!fir.heap<!fir.array<?xf64>>>, !fir.ref<f64>) -> ()
! HLFIR-LABEL: func.func @_QMfpPsolve_defaults(
! HLFIR:         fir.do_loop
! HLFIR-NEXT:      fir.store
! HLFIR-NEXT:      fir.call @__flang_directive.enzyme.fixed_point.1(%{{.*}}, %{{.*}}) {{.*}}{fir.directive = {args = {}, keyword = "fixed_point", prefix = "enzyme"}} : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<32xi8>>) -> ()
! HLFIR-LABEL: func.func @_QMfpPsolve_keywords()
! HLFIR:         fir.call @__flang_directive.enzyme.fixed_point.2(%{{.*}}) {{.*}}{fir.directive = {args = {max_iters = 0 : i64, reduction = -1.500000e-03 : f64}, keyword = "fixed_point", prefix = "enzyme"}} : (!fir.ref<!fir.array<5xf64>>) -> ()
! HLFIR:       func.func private @__flang_directive.enzyme.fixed_point(!fir.ref<!fir.array<5xf64>>, !fir.box<!fir.heap<!fir.array<?xf64>>>, !fir.ref<f64>)

! The plugin's call, its operands computed in front of the loop.
! CHECK-LABEL: define void @_QMfpPsolve()
! CHECK:         call void @llvm.memcpy{{.*}}(ptr {{.*}}%[[DESC:[0-9]+]], ptr {{.*}}@_QMfpEv, i32 48, i1 false)
! CHECK:         %[[ADDRP:[0-9]+]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DESC]], i32 0, i32 0
! CHECK:         %[[ADDR:[0-9]+]] = load ptr, ptr %[[ADDRP]]
! CHECK:         %[[EXTP:[0-9]+]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DESC]], i32 0, i32 7, i64 0, i32 1
! CHECK:         %[[EXT:[0-9]+]] = load i64, ptr %[[EXTP]]
! CHECK:         %[[BYTES:[0-9]+]] = mul i64 %[[EXT]], 8
! CHECK:         br label %[[HEADER:[0-9]+]]
! CHECK:       [[HEADER]]:
! CHECK:         call i32 @step_()
! CHECK:         br i1 %{{.*}}, label %[[BODY:[0-9]+]], label
! CHECK:       [[BODY]]:
! CHECK-NEXT:    call void (double, i64, ptr, ...) @__enzyme_set_fixed_point(double 2.000000e-08, i64 100, ptr @_QMfpPctl, ptr @_QMfpEu, i64 40, ptr %[[ADDR]], i64 %[[BYTES]], ptr getelementptr inbounds nuw (i8, ptr @blk_, i64 24), i64 8)
! CHECK-NEXT:    br label %[[HEADER]]
! CHECK-LABEL: define void @_QMfpPsolve_defaults(ptr {{.*}}%0, ptr {{.*}}%1)
! CHECK:         %[[WEXTP:[0-9]+]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{[0-9]+}}, i32 0, i32 7, i64 0, i32 1
! CHECK:         %[[WEXT:[0-9]+]] = load i64, ptr %[[WEXTP]]
! CHECK:         %[[WBYTES:[0-9]+]] = mul i64 %[[WEXT]], 8
! CHECK:         br label %[[WHEADER:[0-9]+]]
! CHECK:       [[WHEADER]]:
! CHECK:         br i1 %{{.*}}, label %[[WBODY:[0-9]+]], label
! CHECK:       [[WBODY]]:
! CHECK-NEXT:    store i32
! CHECK-NEXT:    call void (double, i64, ptr, ...) @__enzyme_set_fixed_point(double -1.000000e+00, i64 -1, ptr null, ptr %0, i64 %[[WBYTES]], ptr @blk_, i64 32)
! CHECK-LABEL: define void @_QMfpPsolve_keywords()
! CHECK:         call void (double, i64, ptr, ...) @__enzyme_set_fixed_point(double -1.500000e-03, i64 0, ptr null, ptr @_QMfpEu, i64 40)
! CHECK-NOT:   __flang_directive
! CHECK:       declare void @__enzyme_set_fixed_point(double, i64, ptr, ...)
! CHECK-NOT:   __flang_directive

! Optimized, the call stays in the loop, which it is not memory(none) for.
! O2-LABEL: define void @_QMfpPsolve()
! O2:         %[[OADDR:.*]] = load ptr, ptr @_QMfpEv
! O2:         %[[OEXT:.*]] = load i64, ptr getelementptr inbounds nuw (i8, ptr @_QMfpEv, i64 32)
! O2:         %[[OBYTES:.*]] = shl i64 %[[OEXT]], 3
! O2:       [[OBODY:.*]]:
! O2-NEXT:    tail call void (double, i64, ptr, ...) @__enzyme_set_fixed_point(double 2.000000e-08, i64 100, ptr nonnull @_QMfpPctl, ptr nonnull @_QMfpEu, i64 40, ptr %[[OADDR]], i64 %[[OBYTES]], ptr nonnull getelementptr inbounds nuw (i8, ptr @blk_, i64 24), i64 8)
! O2-NEXT:    tail call i32 @step_()
! O2-NEXT:    icmp eq i32
! O2-NEXT:    br i1 %{{.*}}, label %{{.*}}, label %[[OBODY]]
