! The enzyme-fir-type-annotations pass also gives the dummy arguments and
! results of procedures "enzyme_type" attributes, in the encoding Enzyme.jl
! uses for its arguments.
!
! Enzyme's types have no extent: a type at offset -1 holds at every offset.
! The actual argument of an array, CHARACTER or descriptor dummy may be part
! of a larger object (a component, a COMMON block member), so the data of
! such a dummy is typed offset by offset if its size is known (up to
! Enzyme's 500 type offsets), and not at all otherwise.
! -enzyme-fir-arg-unbounded-types types it at every offset instead ([-1,-1],
! as Enzyme.jl does for its arrays), which is unsound for such actuals.
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
! NODATA-LABEL: define void @_QMfir_arg_typesPdescr(
! NODATA-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer,

! OFF-NOT: "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}"

! Module procedures get the attributes also where they are only declared, as
! in a unit that uses the module.
module fir_arg_types
  use, intrinsic :: iso_fortran_env, only: int8, int16, int64, real16, &
                                           real32, real64
  implicit none
  public
  type :: t
    real(real64) :: a
    integer :: n
  end type t

contains

! By reference, a scalar's pointee is laid out from offset 0. A CHARACTER
! dummy is character data, but only its address (%4) is typed: its hidden
! length (%6) is left alone.
! CHECK-LABEL: define void @_QMfir_arg_typesPscalars(
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

! Non-default kinds. An INTEGER or LOGICAL of any kind is Integer (Enzyme's
! Integer has no width, and it has no boolean type), its size is its kind's
! bytes. REAL(2) and REAL(3) are half and bf16; an array of them is laid out
! by the element size of its kind. REAL(10), padded to a size that depends on
! the target, is left alone (REAL(16) would be fp128).
! CHECK-LABEL: define void @_QMfir_arg_typesPkinds(
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %0,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %1,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %2,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %3,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@half}" %4,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@bf16}" %5,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half}" %6,
! CHECK-SAME: ptr noalias %7,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}" %8,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,8]:Integer}" %9,
! CHECK-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half}" %10)
  subroutine kinds(i1, i2, i8, l1, h, b, zh, x10, ai1, al8, ah)
    ! flang's kinds of bfloat16 and of x87 extended precision.
    integer, parameter :: bfloat16 = 3
    integer, parameter :: extended = selected_real_kind(p=18)
    integer(int8), intent(in) :: i1
    integer(int16), intent(in) :: i2
    integer(int64), intent(in) :: i8
    logical(int8), intent(in) :: l1
    real(real16), intent(out) :: h
    real(bfloat16), intent(in) :: b
    complex(real16), intent(in) :: zh
    real(extended), intent(in) :: x10
    integer(int8), intent(in) :: ai1(3)
    logical(int64), intent(in) :: al8(2)
    real(real16), intent(in) :: ah(3)
    h = 1.0_real16
  end subroutine kinds

! An array dummy of a known size is typed offset by offset (b, of unknown
! size, is not).
! CHECK-LABEL: define void @_QMfir_arg_typesParrays(
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %0,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer}" %1,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer}" %2,
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, {{.*}}, [-1,116]:Float@float}" %3)
! UNB-LABEL: define void @_QMfir_arg_typesPscalars(
! UNB-SAME: ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}" %4,
! UNB-LABEL: define void @_QMfir_arg_typesParrays(
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

! A descriptor is typed field by field, and the type of its data only with
! -enzyme-fir-arg-unbounded-types (UNB). A polymorphic dummy (u) is left
! alone.
! UNB-LABEL: define void @_QMfir_arg_typesPdescr(
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, [-1,16]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,32]:Integer, [-1,40]:Integer}" %0,
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, {{.*}}, [-1,64]:Integer}" %1,
! UNB-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, {{[^"]*}}}" %2,
! UNB-SAME: ptr noalias %3)
! CHECK-LABEL: define void @_QMfir_arg_typesPdescr(
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

! By value, a dummy is its scalar type. A derived type (tt) and an
! assumed-rank dummy (r) are left alone.
! CHECK-LABEL: define void @_QMfir_arg_typesPbyvalue(
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

! A function result.
! CHECK: define "enzyme_type"="{[-1]:Float@double}" double @_QMfir_arg_typesPf(ptr noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %0)
  real(real64) function f(x)
    real(real64), intent(in) :: x
    f = x
  end function f
end module fir_arg_types

! External procedures get the attributes only where they are defined: a
! declaration may have been made up from the actual arguments of a call
! through an implicit interface. So user is outside of the module, to be
! defined as an external procedure (user_ is typed), and so is ext, which is
! only declared (ext_ is not typed). The local CHARACTER variable buf is
! character data.
! CHECK-LABEL: define void @user_(
! CHECK-SAME: "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %0)
! CHECK: alloca [64 x i8], i64 1, align 1, !enzyme_type ![[CHR:[0-9]+]]
! NOLOCAL-NOT: alloca [64 x i8], align 1, !enzyme_type
! allow(procedure-not-in-module)
subroutine user(y)
  use, intrinsic :: iso_fortran_env, only: real64
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
