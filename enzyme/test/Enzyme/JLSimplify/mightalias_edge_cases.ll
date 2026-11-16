; RUN: if [ %llvmver -ge 15 ]; then split-file %s %t && \
; RUN:   %opt < %t/nonptr_arg1.ll %OPnewLoadEnzyme -passes="jl-inst-simplify,verify" -S | FileCheck %s --check-prefix=NONPTR && \
; RUN:   %opt < %t/nonint_ret.ll %OPnewLoadEnzyme -passes="jl-inst-simplify,verify" -S | FileCheck %s --check-prefix=NONINT && \
; RUN:   %opt < %t/valid.ll %OPnewLoadEnzyme -passes="jl-inst-simplify,verify" -S | FileCheck %s --check-prefix=VALID; fi

;--- nonptr_arg1.ll
declare i8 @jl_mightalias(ptr, i64)

define i8 @test_nonptr_arg1(i64 %x) {
  %a = alloca i32, align 4
  %r = call i8 @jl_mightalias(ptr %a, i64 %x)
  ret i8 %r
}
; NONPTR-LABEL: define i8 @test_nonptr_arg1(i64 %x)
; NONPTR: %r = call i8 @jl_mightalias(ptr %a, i64 %x)
; NONPTR-NEXT: ret i8 %r

;--- nonint_ret.ll
declare double @jl_mightalias(ptr, ptr)

define double @test_nonint_ret(ptr %arg) {
  %a = alloca i32, align 4
  %r = call double @jl_mightalias(ptr %a, ptr %arg)
  ret double %r
}
; NONINT-LABEL: define double @test_nonint_ret(ptr %arg)
; NONINT: %r = call double @jl_mightalias(ptr %a, ptr %arg)
; NONINT-NEXT: ret double %r

;--- valid.ll
declare i8 @jl_mightalias(ptr, ptr)

define i8 @test_valid_mightalias(ptr %arg) {
  %a = alloca i32, align 4
  %r = call i8 @jl_mightalias(ptr %a, ptr %arg)
  ret i8 %r
}
; VALID-LABEL: define i8 @test_valid_mightalias(ptr %arg)
; VALID: %r = call i8 @jl_mightalias(ptr %a, ptr %arg)
; VALID-NEXT: ret i8 0
