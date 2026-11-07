; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; An alloca outside the entry block is moved to the heap, in another address
; space. With typed pointers, an addrspacecast of it to the heap's address
; space may also change the pointee type, which its users must keep.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-A5"

declare void @use(i8*) "enzyme_inactive"

declare double @__enzyme_autodiff(i8*, ...)

define double @f(double %x) {
entry:
  br label %body

body:
  %a = alloca [5 x i64], align 8, addrspace(5)
  %c = addrspacecast [5 x i64] addrspace(5)* %a to i8*
  call void @use(i8* %c)
  %m = fmul double %x, %x
  ret double %m
}

define double @test(double %x) {
entry:
  %r = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @f to i8*), double %x)
  ret double %r
}

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK: body:
; CHECK-NEXT:   %[[a:.+]] = bitcast i8* %{{.+}} to [5 x i64]*
; CHECK-NEXT:   %[[c:.+]] = bitcast [5 x i64]* %[[a]] to i8*
; CHECK-NEXT:   call void @use(i8* %[[c]])
