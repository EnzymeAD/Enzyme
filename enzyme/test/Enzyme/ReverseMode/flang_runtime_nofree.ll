; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; Calls to LLVM flang runtime functions without a body, which take a pointer
; that is not a known allocation, are inactive but still need a version that
; does not free memory in the reverse pass. These runtime functions do not
; free memory of the program, so they are used as they are.

@sub = private constant [1 x i8] c"x"

; i = INDEX(str, 'x'); FLUSH(6); res = x * x
define double @index(double %x, ptr %str) {
entry:
  %i = call i64 @_FortranAIndex1(ptr %str, i64 8, ptr @sub, i64 1, i1 false)
  call void @_FortranAFlush(i32 6)
  %m = fmul double %x, %x
  ret double %m
}

; PRINT *, x; res = x * x
; The I/O calls are marked inactive here as KnownInactiveFunctions does not
; have them.
define double @print(double %x) {
entry:
  %cookie = call ptr @_FortranAioBeginExternalListOutput(i32 6, ptr null, i32 0) #0
  %ok = call i1 @_FortranAioOutputReal64(ptr %cookie, double %x) #0
  %e = call i32 @_FortranAioEndIoStatement(ptr %cookie) #0
  %m = fmul double %x, %x
  ret double %m
}

declare i64 @_FortranAIndex1(ptr, i64, ptr, i64, i1)
declare void @_FortranAFlush(i32)
declare ptr @_FortranAioBeginExternalListOutput(i32, ptr, i32)
declare i1 @_FortranAioOutputReal64(ptr, double)
declare i32 @_FortranAioEndIoStatement(ptr)

attributes #0 = { "enzyme_inactive" }

define double @dindex(double %x, ptr %str) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @index, double %x, metadata !"enzyme_const", ptr %str)
  ret double %r
}

define double @dprint(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @print, double %x)
  ret double %r
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffeindex(double %x, ptr %str, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %i = call i64 @_FortranAIndex1(ptr %str, i64 8, ptr @sub, i64 1, i1 false)
; CHECK-NEXT:   call void @_FortranAFlush(i32 6)

; CHECK: define internal { double } @diffeprint(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cookie = call ptr @_FortranAioBeginExternalListOutput(i32 6, ptr null, i32 0)
; CHECK-NEXT:   %ok = call i1 @_FortranAioOutputReal64(ptr %cookie, double %x)
; CHECK-NEXT:   %e = call i32 @_FortranAioEndIoStatement(ptr %cookie)
