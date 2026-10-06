! The enzyme-fir-type-annotations pass types local variables on their allocas
! (flang's TBAA names a local but not its type): REAL, INTEGER, LOGICAL and
! COMPLEX scalars and arrays offset by offset if their size is constant and
! within Enzyme's 500 type offsets, at every offset otherwise (an alloca is a
! whole object), and CHARACTER data as Integer throughout. A name that locals
! of different types share is left out.
!
! REQUIRES: flang_enzyme_mlir
! RUN: mkdir -p %t.mod
! RUN: %flang_enzyme -module-dir %t.mod -O0 -emit-llvm %s -o - | FileCheck %s
! RUN: %flang_enzyme -module-dir %t.mod -mmlir -enzyme-fir-local-number-types=false -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=NONUM

! CHECK-LABEL: define void @locs_(
! CHECK-DAG: alloca { double, double }, i64 1, align 8, !enzyme_type ![[Z:[0-9]+]]
! CHECK-DAG: alloca [12 x i8], i64 1, align 1, !enzyme_type ![[CHR:[0-9]+]]
! CHECK-DAG: alloca i32, i64 1, align 4, !enzyme_type ![[I32:[0-9]+]]
! CHECK-DAG: alloca [10 x double], i64 1, align 8, !enzyme_type ![[BUF:[0-9]+]]
! CHECK-DAG: alloca [1000 x double], i64 1, align 8, !enzyme_type ![[ANY:[0-9]+]]
! CHECK-DAG: alloca double, i64 1, align 8, !enzyme_type ![[DBL:[0-9]+]]
! CHECK-DAG: alloca double, i64 %{{[0-9]+}}, align 8, !enzyme_type ![[ANY]]
! NONUM-LABEL: define void @locs_(
! NONUM-NOT: alloca i32, i64 1, align 4, !enzyme_type
! NONUM: alloca [12 x i8], i64 1, align 1, !enzyme_type
subroutine locs(n, x)
  integer :: n, cnt, k
  real(8) :: x(n), acc, buf(10), big(1000), auto(n)
  logical :: flag
  complex(8) :: z
  character(len=12) :: name
  external :: fill, use
  call fill(cnt)
  k = ishft(cnt, 2)
  acc = 0
  do k = 1, n
    acc = acc + x(k)
  end do
  buf = acc; big = acc; auto = acc
  flag = acc > 0; z = acc; name = 'x'
  call use(buf, big, auto, flag, z, name, k)
end subroutine

! CHECK-DAG: ![[Z]] = !{!"Unknown", i32 -1, ![[ZP:[0-9]+]]}
! CHECK-DAG: ![[ZP]] = !{!"Pointer", i32 0, ![[D:[0-9]+]], i32 8, ![[D]]}
! CHECK-DAG: ![[D]] = !{!"Float@double"}
! CHECK-DAG: ![[I32]] = !{!"Unknown", i32 -1, ![[I32P:[0-9]+]]}
! CHECK-DAG: ![[I32P]] = !{!"Pointer", i32 0, ![[INT:[0-9]+]]}
! CHECK-DAG: ![[INT]] = !{!"Integer"}
! CHECK-DAG: ![[BUF]] = !{!"Unknown", i32 -1, ![[BUFP:[0-9]+]]}
! CHECK-DAG: ![[BUFP]] = !{!"Pointer", i32 0, ![[D]], i32 8, ![[D]], i32 16, ![[D]], i32 24, ![[D]], i32 32, ![[D]], i32 40, ![[D]], i32 48, ![[D]], i32 56, ![[D]], i32 64, ![[D]], i32 72, ![[D]]}
! CHECK-DAG: ![[ANY]] = !{!"Unknown", i32 -1, ![[ANYP:[0-9]+]]}
! CHECK-DAG: ![[ANYP]] = !{!"Pointer", i32 -1, ![[D]]}
! CHECK-DAG: ![[CHR]] = !{!"Unknown", i32 -1, ![[CHRP:[0-9]+]]}
! CHECK-DAG: ![[CHRP]] = !{!"Pointer", i32 -1, ![[INT]]}
