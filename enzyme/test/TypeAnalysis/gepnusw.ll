; RUN: if [ %llvmver -ge 19 ]; then %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=f_nusw -S -o /dev/null | FileCheck %s --check-prefix=NUSW; fi
; RUN: if [ %llvmver -ge 19 ]; then %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=f_inbounds -S -o /dev/null | FileCheck %s --check-prefix=INBOUNDS; fi

; A `nusw` GEP must type its indices exactly as an `inbounds` one does. The two
; functions below are identical apart from the no-wrap flag.

declare i32 @llvm.smax.i32(i32, i32)

define double @f_nusw(ptr %yh, ptr %ldyh, i64 %i) {
entry:
  %n = load i32, ptr %ldyh, align 4
  %m = call i32 @llvm.smax.i32(i32 %n, i32 0)
  %z = zext i32 %m to i64
  %idx = mul i64 %i, %z
  %p = getelementptr nusw nuw double, ptr %yh, i64 %idx
  %v = load double, ptr %p, align 8
  ret double %v
}

define double @f_inbounds(ptr %yh, ptr %ldyh, i64 %i) {
entry:
  %n = load i32, ptr %ldyh, align 4
  %m = call i32 @llvm.smax.i32(i32 %n, i32 0)
  %z = zext i32 %m to i64
  %idx = mul i64 %i, %z
  %p = getelementptr inbounds double, ptr %yh, i64 %idx
  %v = load double, ptr %p, align 8
  ret double %v
}

; NUSW: ptr %yh: {[-1]:Pointer}
; NUSW-NEXT: ptr %ldyh: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; NUSW-NEXT: i64 %i: {[-1]:Integer}
; NUSW-NEXT: entry
; NUSW-NEXT:   %n = load i32, ptr %ldyh, align 4: {[-1]:Integer}
; NUSW-NEXT:   %m = call i32 @llvm.smax.i32(i32 %n, i32 0): {[-1]:Integer}
; NUSW-NEXT:   %z = zext i32 %m to i64: {[-1]:Integer}
; NUSW-NEXT:   %idx = mul i64 %i, %z: {[-1]:Integer}
; NUSW-NEXT:   %p = getelementptr nusw nuw double, ptr %yh, i64 %idx: {[-1]:Pointer, [-1,0]:Float@double}
; NUSW-NEXT:   %v = load double, ptr %p, align 8: {[-1]:Float@double}

; INBOUNDS: ptr %yh: {[-1]:Pointer}
; INBOUNDS-NEXT: ptr %ldyh: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; INBOUNDS-NEXT: i64 %i: {[-1]:Integer}
; INBOUNDS-NEXT: entry
; INBOUNDS-NEXT:   %n = load i32, ptr %ldyh, align 4: {[-1]:Integer}
; INBOUNDS-NEXT:   %m = call i32 @llvm.smax.i32(i32 %n, i32 0): {[-1]:Integer}
; INBOUNDS-NEXT:   %z = zext i32 %m to i64: {[-1]:Integer}
; INBOUNDS-NEXT:   %idx = mul i64 %i, %z: {[-1]:Integer}
; INBOUNDS-NEXT:   %p = getelementptr inbounds double, ptr %yh, i64 %idx: {[-1]:Pointer, [-1,0]:Float@double}
; INBOUNDS-NEXT:   %v = load double, ptr %p, align 8: {[-1]:Float@double}
