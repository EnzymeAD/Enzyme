; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -opaque-pointers=1 -type-analysis-func=foo -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -opaque-pointers=1 -type-analysis-func=foo -S -o /dev/null | FileCheck %s

; The CHARACTER arguments (pointer + length) of the LLVM flang I/O runtime
; point to characters: exactly `len` bytes (times the kind for
; OutputCharacter) are typed as Integer. A length that is not a constant
; types only the pointer. %item is a CHARACTER(8) component followed by a
; pointer component, which must not conflict.

declare ptr @_FortranAioBeginExternalFormattedOutput(ptr, i64, ptr, i32, ptr, i32)
declare ptr @_FortranAioBeginInternalFormattedOutput(ptr, i64, ptr, i64, ptr, ptr, i64, ptr, i32)
declare i1 @_FortranAioOutputAscii(ptr, ptr, i64)
declare i1 @_FortranAioOutputCharacter(ptr, ptr, i64, i32)
declare i1 @_FortranAioSetAdvance(ptr, ptr, i64)

define void @foo(ptr %fmt, ptr %unit, ptr %ifmt, ptr %str, ptr %wide, ptr %adv, ptr %dyn, i64 %n, ptr %item) {
entry:
  %c1 = call ptr @_FortranAioBeginExternalFormattedOutput(ptr %fmt, i64 4, ptr null, i32 6, ptr null, i32 1)
  %c2 = call ptr @_FortranAioBeginInternalFormattedOutput(ptr %unit, i64 3, ptr %ifmt, i64 2, ptr null, ptr null, i64 0, ptr null, i32 1)
  %r1 = call i1 @_FortranAioOutputAscii(ptr %c1, ptr %str, i64 3)
  %r2 = call i1 @_FortranAioOutputCharacter(ptr %c1, ptr %wide, i64 2, i32 4)
  %r3 = call i1 @_FortranAioSetAdvance(ptr %c1, ptr %adv, i64 2)
  %r4 = call i1 @_FortranAioOutputAscii(ptr %c1, ptr %dyn, i64 %n)
  %r5 = call i1 @_FortranAioOutputAscii(ptr %c1, ptr %item, i64 8)
  %gep = getelementptr inbounds i8, ptr %item, i64 8
  %p = load ptr, ptr %gep, align 8
  %d = load double, ptr %p, align 8
  ret void
}

; CHECK: foo - {} |
; CHECK-NEXT: ptr %fmt: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; CHECK-NEXT: ptr %unit: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}
; CHECK-NEXT: ptr %ifmt: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer}
; CHECK-NEXT: ptr %str: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}
; CHECK-NEXT: ptr %wide: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: ptr %adv: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer}
; CHECK-NEXT: ptr %dyn: {[-1]:Pointer}
; CHECK-NEXT: i64 %n: {[-1]:Integer}
; CHECK-NEXT: ptr %item: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer}
