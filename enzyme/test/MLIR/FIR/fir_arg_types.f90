! The enzyme-fir-type-annotations pass also gives the dummy arguments and
! results of procedures "enzyme_type" attributes, in the encoding Enzyme.jl
! uses: by reference, the pointee laid out from offset 0; a descriptor field
! by field; by value, the scalar. Enzyme's types have no extent, and the
! actual argument of an array, CHARACTER or descriptor dummy may be part of a
! larger object (a component, a COMMON block member), so their data is typed
! offset by offset if the size is known (up to Enzyme's 500 type offsets),
! and not at all otherwise; -enzyme-fir-arg-unbounded-types types it at every
! offset ([-1,-1], as Enzyme.jl does for its arrays).
! Module procedures get them also where only declared (as in a unit that uses
! the module), external procedures only where defined (a declaration may come
! from an implicit interface). A CHARACTER dummy is character data; its
! hidden length is left alone (typed Integer, Enzyme took a mask of it for a
! possibly floating-point operation). Polymorphic, assumed-type and
! assumed-rank dummies and derived types are left alone. Local CHARACTER
! variables are character data too.
!
! REQUIRES: flang_enzyme_mlir
! RUN: mkdir -p %t.mod
! RUN: %flang_enzyme -module-dir %t.mod -O0 -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_enzyme -module-dir %t.mod -mmlir -enzyme-fir-arg-types=false -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF
! The other annotations off, these on:
! RUN: %flang_enzyme -module-dir %t.mod -mmlir -enzyme-fir-common-types=false -mmlir -enzyme-fir-runtime-types=false -mmlir -enzyme-fir-literal-types=false -O0 -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_enzyme -module-dir %t.mod -mmlir -enzyme-fir-local-types=false -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=NOLOCAL
! RUN: %flang_enzyme -module-dir %t.mod -mmlir -enzyme-fir-arg-unbounded-types -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=UNB
! RUN: %flang_enzyme -module-dir %t.mod -mmlir -enzyme-fir-arg-unbounded-types -mmlir -enzyme-fir-arg-descriptor-data-types=false -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=NODATA
! NODATA-LABEL: define void @_QMmPdescr(
! NODATA-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer,

! OFF-NOT: "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}"

module m
  use, intrinsic :: iso_fortran_env, only: real32, real64
  implicit none
  public
  type :: t
    real(real64) :: a
    integer :: n
  end type t
contains
! CHECK-LABEL: define void @_QMmPscalars(
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %0,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float}" %1,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %2,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %3,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer}" %4,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}" %5,
! CHECK-SAME: i64 %6)
  subroutine scalars(x8, x4, i, l, c, z)
    real(real64), intent(out) :: x8
    real(real32), intent(in) :: x4
    integer, intent(in) :: i
    logical, intent(in) :: l
    character(len=*), intent(in) :: c
    complex(real64), intent(in) :: z
    x8 = x4 + i
  end subroutine scalars

! CHECK-LABEL: define void @_QMmParrays(
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %0,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer}" %1,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer}" %2,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, {{.*}}, [-1,116]:Float@float}" %3)
! UNB-LABEL: define void @_QMmPscalars(
! UNB-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}" %4,
! UNB-LABEL: define void @_QMmParrays(
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %1,
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %2,
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@float}" %3)
  subroutine arrays(n, a, b, s)
    integer, intent(in) :: n
    real(real64), intent(out) :: a(n)
    ! allow(assumed-size)
    real(real64), intent(in) :: b(*)
    real(real32), intent(in) :: s(10, 3)
    a(1) = b(1) + s(1, 1)
  end subroutine arrays

! UNB-LABEL: define void @_QMmPdescr(
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, [-1,16]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,32]:Integer, [-1,40]:Integer}" %0,
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, {{.*}}, [-1,64]:Integer}" %1,
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, {{[^"]*}}}" %2,
! UNB-SAME: ptr noalias %3)
! CHECK-LABEL: define void @_QMmPdescr(
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer, [-1,16]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,32]:Integer, [-1,40]:Integer}" %0,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer, {{.*}}, [-1,64]:Integer}" %1,
! CHECK-SAME: ptr noalias %3)
  subroutine descr(a, p, q, u)
    real(real64), intent(inout) :: a(:)
    real(real64), allocatable, intent(in) :: p(:,:)
    real(real32), pointer, intent(in) :: q(:)
    class(*), intent(in) :: u
    a(1) = 1
  end subroutine descr

! CHECK-LABEL: define void @_QMmPbyvalue(
! CHECK-SAME: double "enzyme_type"="{[-1]:Float@double}" %0,
! CHECK-SAME: i32 "enzyme_type"="{[-1]:Integer}" %1,
! CHECK-SAME: ptr noalias %2, ptr noalias %3)
  subroutine byvalue(x, n, tt, r)
    real(real64), value :: x
    integer, value :: n
    type(t), intent(inout) :: tt
    integer, intent(in) :: r(..)
    tt%a = x
  end subroutine byvalue

! CHECK: define "enzyme_type"="{[-1]:Float@double}" double @_QMmPf(ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %0)
  real(real64) function f(x)
    real(real64), intent(in) :: x
    f = x
  end function f
end module m

! CHECK-LABEL: define void @user_(
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %0)
! CHECK: alloca [64 x i8], i64 1, align 1, !enzyme_type ![[CHR:[0-9]+]]
! NOLOCAL-NOT: alloca [64 x i8], align 1, !enzyme_type
! allow(procedure-not-in-module)
subroutine user(y)
  use, intrinsic :: iso_fortran_env, only: real64
  use m, only: t
  implicit none
  real(real64), intent(inout) :: y
  character(len=64) :: buf
  external :: ext
  buf = "x"
  call ext(y, 1.0, buf)
end subroutine user
! A name that a local of another type has too is left out (here the second
! BLOCK's v):
! CHECK-LABEL: define void @collide_(
! CHECK: alloca [8 x i8], i64 1, align 1{{$}}
! CHECK: alloca [8 x i8], i64 1, align 1, !enzyme_type ![[CHR]]
! allow(procedure-not-in-module)
subroutine collide()
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  character(len=8) :: w
  w = "b"
  print *, w
  block
    character(len=8) :: v
    v = "a"
    print *, v
  end block
  block
    real(real64) :: v(4)
    v = 1
    print *, v
  end block
end subroutine collide

! CHECK: declare void @ext_(ptr, ptr, ptr, i64)
! CHECK: ![[CHR]] = !{!"Unknown", i32 -1, ![[CHRP:[0-9]+]]}
! CHECK: ![[CHRP]] = !{!"Pointer", i32 -1, ![[INT:[0-9]+]]}

