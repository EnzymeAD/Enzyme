! The enzyme-fir-type-annotations pass also gives the dummy arguments and
! results of procedures "enzyme_type" attributes, in the encoding Enzyme.jl
! uses: by reference, the pointee laid out from offset 0 (a scalar) or at
! every offset (an array); a descriptor field by field; by value, the scalar.
! Module procedures get them also where only declared (as in a unit that uses
! the module), external procedures only where defined (a declaration may come
! from an implicit interface). Polymorphic, assumed-type and assumed-rank
! dummies, derived types and CHARACTER descriptors are left alone.
!
! REQUIRES: flang_directives
! RUN: %fc -fc1 %flangFc1Directives -O0 -emit-llvm %s -o - | FileCheck %s
! RUN: %fc -fc1 %flangFc1Directives -mmlir -enzyme-fir-arg-types=false -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

! OFF-NOT: "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}"

module m
  implicit none
  type :: t
    real(8) :: a
    integer :: n
  end type
contains
! CHECK-LABEL: define void @_QMmPscalars(
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %0,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float}" %1,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %2,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %3,
! CHECK-SAME: ptr noalias %4,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}" %5,
! CHECK-SAME: i64 %6)
  subroutine scalars(x8, x4, i, l, c, z)
    real(8) :: x8
    real(4) :: x4
    integer :: i
    logical :: l
    character(len=*) :: c
    complex(8) :: z
    x8 = x4 + i
  end subroutine

! CHECK-LABEL: define void @_QMmParrays(
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %0,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %1,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %2,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@float}" %3)
  subroutine arrays(n, a, b, s)
    integer :: n
    real(8) :: a(n), b(*)
    real(4) :: s(10, 3)
    a(1) = b(1) + s(1, 1)
  end subroutine

! CHECK-LABEL: define void @_QMmPdescr(
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, [-1,16]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,32]:Integer, [-1,40]:Integer}" %0,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, {{.*}}, [-1,64]:Integer}" %1,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, {{[^"]*}}}" %2,
! CHECK-SAME: ptr noalias %3)
  subroutine descr(a, p, q, u)
    real(8) :: a(:)
    real(8), allocatable :: p(:,:)
    real(4), pointer :: q(:)
    class(*) :: u
    a(1) = 1
  end subroutine

! CHECK-LABEL: define void @_QMmPbyvalue(
! CHECK-SAME: double "enzyme_type"="{[-1]:Float@double}" %0,
! CHECK-SAME: i32 "enzyme_type"="{[-1]:Integer}" %1,
! CHECK-SAME: ptr noalias %2, ptr noalias %3)
  subroutine byvalue(x, n, tt, r)
    real(8), value :: x
    integer, value :: n
    type(t) :: tt
    integer :: r(..)
    tt%a = x
  end subroutine

! CHECK: define "enzyme_type"="{[-1]:Float@double}" double @_QMmPf(ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %0)
  real(8) function f(x)
    real(8) :: x
    f = x
  end function
end module

! CHECK-LABEL: define void @user_(
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %0)
subroutine user(y)
  use m, only: t
  real(8) :: y
  external :: ext
  call ext(y, 1.0)
end subroutine

! CHECK-DAG: declare void @ext_(ptr, ptr)
