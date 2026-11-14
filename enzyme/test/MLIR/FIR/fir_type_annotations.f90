! The FlangEnzymeMLIR plugin's enzyme-fir-type-annotations pass carries FIR
! types that LLVM IR erases to LLVM Enzyme's type analysis, as !enzyme_type
! metadata and "enzyme_type" attributes.
!
! REQUIRES: flang_enzyme_mlir
! RUN: mkdir -p %t.mod
! RUN: %flang_enzyme -module-dir %t.mod -O0 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,O0
! RUN: %flang_enzyme -module-dir %t.mod -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
! RUN: %flang_enzyme -module-dir %t.mod -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=RT

! COMMON blocks: the type at each member offset, if the declarations lay out
! all of the block and agree.
!
! /state/ is a REAL*8 and an INTEGER:
! CHECK-DAG: @state_ = {{.*}}global [12 x i8] {{.*}}!enzyme_type ![[STATE:[0-9]+]]
! CHECK-DAG: ![[STATE]] = !{!"Unknown", i32 -1, ![[STATEP:[0-9]+]]}
! CHECK-DAG: ![[STATEP]] = !{!"Pointer", i32 0, ![[DBL:[0-9]+]], i32 8, ![[INT:[0-9]+]]}
! CHECK-DAG: ![[DBL]] = !{!"Float@double"}
! CHECK-DAG: ![[INT]] = !{!"Integer"}
!
! /mixed/ is REAL at offset 0 in one subroutine and INTEGER in the other: it
! stays unknown.
! CHECK-DAG: @mixed_ = {{.*}}global [8 x i8] zeroinitializer, align 4{{$}}
!
! /bufs/ is larger than the offsets Enzyme's type analysis keeps (500 bytes):
! it stays unknown, else Enzyme would take REAL*8, the type it keeps for the
! first 500 bytes, for all of the block.
! CHECK-DAG: @bufs_ = {{.*}}global [196608 x i8] zeroinitializer, align 8{{$}}
!
! A block of a single scalar type is that type at every offset, whatever its
! size: here REAL*8 arrays, and CHARACTER data (Integer bytes, /names/ below,
! with the literals).
! CHECK-DAG: @fields_ = {{.*}}global [16000 x i8] {{.*}}!enzyme_type ![[FIELDS:[0-9]+]]
! CHECK-DAG: ![[FIELDS]] = !{!"Unknown", i32 -1, ![[FIELDSP:[0-9]+]]}
! CHECK-DAG: ![[FIELDSP]] = !{!"Pointer", i32 -1, ![[DBL]]}
! allow(procedure-not-in-module)
subroutine uses_common(x)
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  real(real64), intent(in) :: x
  real(real64) :: a
  integer :: n
  ! allow(common-block)
  common /state/ a, n
  real :: r, s
  ! allow(common-block)
  common /mixed/ r, s
  a = x
  n = 1
  r = 1.0
  s = 2.0
end subroutine uses_common

! allow(procedure-not-in-module)
subroutine buffers()
  use, intrinsic :: iso_fortran_env, only: real32, real64
  implicit none
  real(real64) :: b8(16384)
  real(real32) :: b4(16384)
  ! allow(common-block)
  common /bufs/ b8, b4
  b8(1) = 1.0
  b4(1) = 1.0
end subroutine buffers

! allow(procedure-not-in-module)
subroutine uniform_blocks()
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  character(len=512) :: fname(3), title
  ! allow(common-block)
  common /names/ fname, title
  real(real64) :: u(1000), v(1000)
  ! allow(common-block)
  common /fields/ u, v
  fname(1) = "a"
  u(1) = 1.0
end subroutine uniform_blocks

! allow(procedure-not-in-module)
subroutine other_view()
  implicit none
  integer :: i
  real :: s
  ! allow(common-block)
  common /mixed/ i, s
  i = 2
end subroutine other_view

! Character literals are character data.
! CHECK-DAG: @_QQcl{{.*}} = {{.*}}constant [{{[0-9]+}} x i8] {{.*}}!enzyme_type ![[CHARS:[0-9]+]]
! CHECK-DAG: ![[CHARS]] = !{!"Unknown", i32 -1, ![[CHARSP:[0-9]+]]}
! CHECK-DAG: ![[CHARSP]] = !{!"Pointer", i32 -1, ![[INT]]}
! CHECK-DAG: @names_ = {{.*}}global [2048 x i8] {{.*}}!enzyme_type ![[CHARS]]{{$}}

! Runtime calls: what the conversions for the call erased. A descriptor is
! typed field by field, with the type of its data if that is a whole object
! (here a local ALLOCATABLE); character data is Integer; the I/O cookie (an
! opaque pointer of its own type) is left alone. `a = b` is an assignment to
! a whole ALLOCATABLE, which allocates a here; flang calls _FortranAAssign for
! it. `a(:) = b` would not allocate a, and a is not allocated.
! O0-DAG: call void @_FortranAAssign{{[A-Za-z]*}}(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, [-1,8]:Integer, [-1,16]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,32]:Integer, [-1,40]:Integer}"
! CHECK-DAG: call {{.*}}@_FortranAioOutputAscii(ptr %{{[0-9]+}}, ptr {{(nonnull )?}}"enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}"
! allow(procedure-not-in-module)
subroutine copy(b, n)
  implicit none
  real, intent(in) :: b(:)
  integer, intent(in) :: n
  real, allocatable :: a(:)
  character(len=16) :: name
  name = "copy"
  a = b
  print *, name, a(n)
end subroutine copy

! Data that may be part of a larger object of other types (a member of a
! COMMON block or EQUIVALENCE group, a component, a dummy argument) keeps its
! descriptor layout but not the type of its data: Enzyme's types have no
! extent.
! RT-LABEL: define void @read_common_
! RT: call {{.*}}@_FortranAioInputDescriptor(ptr %{{[0-9]+}}, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer,
! RT: call {{.*}}@_FortranAioInputDescriptor(ptr %{{[0-9]+}}, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, [-1,8]:Integer,
! RT: call {{.*}}@_FortranAioInputDescriptor(ptr %{{[0-9]+}}, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer,
! RT: call {{.*}}@_FortranAioInputDescriptor(ptr %{{[0-9]+}}, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer,
! allow(procedure-not-in-module)
subroutine read_common(u, n, d)
  use, intrinsic :: iso_fortran_env, only: real32, real64
  implicit none
  type :: t
    character(len=8) :: name
    real(real32) :: x(4)
  end type t
  integer, intent(in) :: u, n
  integer :: i
  real(real32), intent(out) :: d(4)
  real(real64) :: c8(4)
  real(real32) :: c4(4)
  ! allow(common-block)
  common /rbufs/ c8, c4
  real(real32) :: loc(4)
  type(t) :: v
  read(u) (c4(i), i=1,n)
  read(u) loc
  read(u) d
  read(u) v%x
end subroutine read_common
