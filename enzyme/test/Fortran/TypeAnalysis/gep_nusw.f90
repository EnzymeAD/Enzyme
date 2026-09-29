! REQUIRES: flangenzyme
! RUN: %fc -S -emit-llvm -O1 %s -o %t.ll
! RUN: FileCheck %s --check-prefix=IR < %t.ll
! RUN: %opt < %t.ll %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=stride_index_ -S -o /dev/null | FileCheck %s --check-prefix=TA

! Companion to test/TypeAnalysis/gepnusw.ll: shows that the `nusw nuw` GEP the
! rule is about is what flang actually emits for an array element address.

subroutine stride_index(a, n, i, res)
  implicit none
  integer, intent(in) :: n, i
  double precision, intent(in) :: a(n, *)
  double precision, intent(out) :: res
  res = a(i, 2)
end subroutine stride_index

! IR: getelementptr nusw nuw

! The leading dimension `n`, and the stride computation indexing with it:
! TA: ptr %{{[0-9]+}}: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
! TA: %{{[0-9]+}} = load i32, ptr %{{[0-9]+}}, align 4{{.*}}: {[-1]:Integer}
! TA-NEXT: %{{[0-9]+}} = tail call i32 @llvm.smax.i32({{.*}}): {[-1]:Integer}
! TA-NEXT: %{{[0-9]+}} = zext nneg i32 %{{[0-9]+}} to i64: {[-1]:Integer}
