! The FlangEnzymeMLIR plugin's enzyme-fir-type-annotations pass carries FIR
! types that LLVM IR erases to LLVM Enzyme's type analysis, as !enzyme_type
! metadata and "enzyme_type" attributes.
!
! REQUIRES: flang_directives
! RUN: %fc -fc1 %flangFc1Directives -O0 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,O0
! RUN: %fc -fc1 %flangFc1Directives -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK

! COMMON blocks: the type at each member offset, if the declares lay out all
! of the block and agree. /mixed/ is real at offset 0 in one subroutine and
! integer in the other, and /bufs/ is larger than the offsets Enzyme's type
! analysis keeps (500 bytes): they stay unknown (Enzyme would take REAL*8,
! the type it keeps for /bufs/, for all of the block).
! CHECK-DAG: @state_ = {{.*}}global [12 x i8] {{.*}}!enzyme_type ![[STATE:[0-9]+]]
! CHECK-DAG: ![[STATE]] = !{!"Unknown", i32 -1, ![[STATEP:[0-9]+]]}
! CHECK-DAG: ![[STATEP]] = !{!"Pointer", i32 0, ![[DBL:[0-9]+]], i32 8, ![[INT:[0-9]+]]}
! CHECK-DAG: ![[DBL]] = !{!"Float@double"}
! CHECK-DAG: ![[INT]] = !{!"Integer"}
! CHECK-DAG: @mixed_ = {{.*}}global [8 x i8] zeroinitializer, align 4{{$}}
! CHECK-DAG: @bufs_ = {{.*}}global [196608 x i8] zeroinitializer, align 8{{$}}
! Blocks of a single scalar type are that type at every offset, whatever
! their size, e.g. CHARACTER data (Integer bytes) and REAL*8 arrays:
! (/names/ below, as the literals)
! CHECK-DAG: @fields_ = {{.*}}global [16000 x i8] {{.*}}!enzyme_type ![[FIELDS:[0-9]+]]
! CHECK-DAG: ![[FIELDS]] = !{!"Unknown", i32 -1, ![[FIELDSP:[0-9]+]]}
! CHECK-DAG: ![[FIELDSP]] = !{!"Pointer", i32 -1, ![[DBL]]}
subroutine uses_common(x)
  real(8) :: x, a
  integer :: n
  common /state/ a, n
  real :: r, s
  common /mixed/ r, s
  a = x
  n = 1
  r = 1.0
  s = 2.0
end subroutine

subroutine buffers()
  real(8) :: b8(16384)
  real(4) :: b4(16384)
  common /bufs/ b8, b4
  b8(1) = 1.0
  b4(1) = 1.0
end subroutine

subroutine uniform_blocks()
  character(len=512) :: fname(3), title
  common /names/ fname, title
  real(8) :: u(1000), v(1000)
  common /fields/ u, v
  fname(1) = 'a'
  u(1) = 1.0
end subroutine

subroutine other_view()
  integer :: i
  real :: s
  common /mixed/ i, s
  i = 2
end subroutine

! Character literals are character data.
! CHECK-DAG: @_QQcl{{.*}} = {{.*}}constant [{{[0-9]+}} x i8] {{.*}}!enzyme_type ![[CHARS:[0-9]+]]
! CHECK-DAG: ![[CHARS]] = !{!"Unknown", i32 -1, ![[CHARSP:[0-9]+]]}
! CHECK-DAG: ![[CHARSP]] = !{!"Pointer", i32 -1, ![[INT]]}
! CHECK-DAG: @names_ = {{.*}}global [2048 x i8] {{.*}}!enzyme_type ![[CHARS]]{{$}}

! Runtime calls: what the conversions for the call erased. A descriptor is
! typed field by field, with its element type; character data is Integer;
! the I/O cookie (an opaque pointer of its own type) is left alone.
! O0-DAG: call void @_FortranAAssign{{[A-Za-z]*}}(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, [-1,8]:Integer, [-1,16]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,32]:Integer, [-1,40]:Integer}"
! CHECK-DAG: call {{.*}}@_FortranAioOutputAscii(ptr %{{[0-9]+}}, ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}"
subroutine copy(a, b, name)
  real, allocatable :: a(:)
  real, intent(in) :: b(:)
  character(len=*), intent(in) :: name
  a = b
  print *, name, a(1)
end subroutine
