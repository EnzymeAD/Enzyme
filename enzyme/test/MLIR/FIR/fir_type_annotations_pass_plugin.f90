! The LLVM pass plugin alone, `flang -fpass-plugin=FlangEnzyme-<v>.so` with no
! -load, brings in the FIR type annotations of FlangEnzymeMLIR (it loads
! FlangEnzymeMLIR, without its differentiation passes), and their -mmlir
! switches, which parse because flang loads pass plugins before it parses
! -mmlir. Loading FlangEnzymeMLIR as well loads it once.
!
! REQUIRES: flang_enzyme_mlir, flangenzyme
! RUN: %flang_fc1 %loadFlangEnzyme -O0 -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_fc1 %loadFlangEnzyme -mmlir -enzyme-fir-common-types=false -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=NOCOMMON
! RUN: %flang_fc1 %loadFlangEnzyme -mmlir -enzyme-fir-type-annotations=false -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF
! RUN: %flang_enzyme %loadFlangEnzyme -O0 -emit-llvm %s -o - | FileCheck %s

! CHECK-DAG: @state_ = {{.*}}global [12 x i8] {{.*}}!enzyme_type ![[STATE:[0-9]+]]
! CHECK-DAG: ![[STATE]] = !{!"Unknown", i32 -1, ![[STATEP:[0-9]+]]}
! CHECK-DAG: ![[STATEP]] = !{!"Pointer", i32 0, ![[DBL:[0-9]+]], i32 8, ![[INT:[0-9]+]]}
! CHECK-DAG: ![[DBL]] = !{!"Float@double"}
! CHECK-DAG: ![[INT]] = !{!"Integer"}
! NOCOMMON: @state_ = {{.*}}global [12 x i8] zeroinitializer, align {{[0-9]+}}{{$}}
! OFF-NOT: enzyme_type

! allow(procedure-not-in-module)
subroutine uses_common(x)
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  real(real64), intent(out) :: x
  real(real64) :: a
  integer :: n
  ! allow(common-block)
  common /state/ a, n
  x = a * n
end subroutine uses_common
