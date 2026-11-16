; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="simple-gvn,verify" -S | FileCheck %s; fi

target datalayout = "E-m:e-p:64:64-i64:64"

; 0x12345678 stored in big-endian has bytes: [0]=0x12 (18), [1]=0x34 (52), [2]=0x56 (86), [3]=0x78 (120).
define i8 @test_be_byte0(ptr noalias %p) {
entry:
  store i32 305419896, ptr %p, align 4
  %v = load i8, ptr %p, align 1
  ret i8 %v
}

define i8 @test_be_byte1(ptr noalias %p) {
entry:
  store i32 305419896, ptr %p, align 4
  %g = getelementptr i8, ptr %p, i64 1
  %v = load i8, ptr %g, align 1
  ret i8 %v
}

define i8 @test_be_byte3(ptr noalias %p) {
entry:
  store i32 305419896, ptr %p, align 4
  %g = getelementptr i8, ptr %p, i64 3
  %v = load i8, ptr %g, align 1
  ret i8 %v
}

define i16 @test_be_hi16(ptr noalias %p, i32 %x) {
entry:
  store i32 %x, ptr %p, align 4
  %v = load i16, ptr %p, align 2
  ret i16 %v
}

; CHECK-LABEL: define i8 @test_be_byte0(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 305419896, ptr %p, align 4
; CHECK-NEXT:   ret i8 18

; CHECK-LABEL: define i8 @test_be_byte1(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 305419896, ptr %p, align 4
; CHECK-NEXT:   %g = getelementptr i8, ptr %p, i64 1
; CHECK-NEXT:   ret i8 52

; CHECK-LABEL: define i8 @test_be_byte3(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 305419896, ptr %p, align 4
; CHECK-NEXT:   %g = getelementptr i8, ptr %p, i64 3
; CHECK-NEXT:   ret i8 120

; CHECK-LABEL: define i16 @test_be_hi16(ptr noalias %p, i32 %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 %x, ptr %p, align 4
; CHECK-NEXT:   %0 = lshr i32 %x, 16
; CHECK-NEXT:   %1 = trunc i32 %0 to i16
; CHECK-NEXT:   ret i16 %1
