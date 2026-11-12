; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=null_ce_gep -o /dev/null | FileCheck %s --check-prefix=NULLGEP; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=null_ce_gep -S -o /dev/null | FileCheck %s --check-prefix=NULLGEP
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -enzyme-strict-aliasing=0 -type-analysis-func=noninbounds_ce_gep -o /dev/null | FileCheck %s --check-prefix=NONINBOUNDS; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -enzyme-strict-aliasing=0 -type-analysis-func=noninbounds_ce_gep -S -o /dev/null | FileCheck %s --check-prefix=NONINBOUNDS

@g = external global i8
@h = external global i8

define void @null_ce_gep(ptr %out) {
entry:
  store ptr getelementptr inbounds (i8, ptr null, i64 sub (i64 ptrtoint (ptr @g to i64), i64 ptrtoint (ptr @h to i64))), ptr %out, align 8
  ret void
}

define void @noninbounds_ce_gep(ptr %out) {
entry:
  store ptr getelementptr (i8, ptr @g, i64 sub (i64 ptrtoint (ptr @g to i64), i64 ptrtoint (ptr @h to i64))), ptr %out, align 8
  ret void
}

; NULLGEP: null_ce_gep - {} |{[-1]:Pointer}:{}
; NULLGEP: ptr %out: {[-1]:Pointer}

; NONINBOUNDS: noninbounds_ce_gep - {} |{[-1]:Pointer}:{}
; NONINBOUNDS: ptr %out: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer}
