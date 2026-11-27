; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=partial_int_store -S -o /dev/null | FileCheck %s --check-prefix=STORE; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=int_and_anything_store -S -o /dev/null | FileCheck %s --check-prefix=ANYTHING; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=partial_int_atomic -S -o /dev/null | FileCheck %s --check-prefix=ATOMIC; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=partial_int_cmpxchg -S -o /dev/null | FileCheck %s --check-prefix=CMPXCHG; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=full_int_store -S -o /dev/null | FileCheck %s --check-prefix=FULLINT; fi

; Bytes 0..3 of %buf are Integer while bytes 4..7 are Unknown; an 8-byte store
; cannot be treated as all-integral memory.
define void @partial_int_store(ptr %in, ptr %out) {
entry:
  %buf = alloca i64, align 8
  store i32 42, ptr %buf, align 4
  %v = load i64, ptr %in, align 8
  store i64 %v, ptr %buf, align 8
  %v2 = load i64, ptr %buf, align 8
  store i64 %v2, ptr %out, align 8
  ret void
}

; Bytes 0..3 are Integer while bytes 4..7 are Anything (from float 0.0); an
; 8-byte store must remain active because Anything may hold floating-point data.
define void @int_and_anything_store(ptr %in, ptr %out) {
entry:
  %buf = alloca { i32, float }, align 8
  store { i32, float } { i32 42, float 0.0 }, ptr %buf, align 8
  %v = load i64, ptr %in, align 8
  store i64 %v, ptr %buf, align 8
  %v2 = load i64, ptr %buf, align 8
  store i64 %v2, ptr %out, align 8
  ret void
}

; Same partial-integral bug via atomicrmw xchg.
define void @partial_int_atomic(ptr %in, ptr %out) {
entry:
  %buf = alloca i64, align 8
  store i32 42, ptr %buf, align 4
  %v = load i64, ptr %in, align 8
  %old = atomicrmw xchg ptr %buf, i64 %v monotonic, align 8
  %v2 = load i64, ptr %buf, align 8
  store i64 %v2, ptr %out, align 8
  ret void
}

; Same partial-integral bug via cmpxchg.
define void @partial_int_cmpxchg(ptr %in, ptr %out) {
entry:
  %buf = alloca i64, align 8
  store i32 42, ptr %buf, align 4
  %v = load i64, ptr %in, align 8
  %res = cmpxchg ptr %buf, i64 0, i64 %v monotonic monotonic, align 8
  %v2 = load i64, ptr %buf, align 8
  store i64 %v2, ptr %out, align 8
  ret void
}

; When all 8 bytes are known Integer, the 8-byte store remains inactive.
define void @full_int_store(ptr %in, ptr %out) {
entry:
  %buf = alloca i64, align 8
  store i64 42, ptr %buf, align 8
  %v = load i64, ptr %in, align 8
  store i64 %v, ptr %buf, align 8
  %v2 = load i64, ptr %buf, align 8
  store i64 %v2, ptr %out, align 8
  ret void
}

; STORE: store i64 %v, ptr %buf, align 8: icv:1 ici:0
; STORE: store i64 %v2, ptr %out, align 8: icv:1 ici:0

; ANYTHING: store i64 %v, ptr %buf, align 8: icv:1 ici:0
; ANYTHING: store i64 %v2, ptr %out, align 8: icv:1 ici:0

; ATOMIC: %old = atomicrmw xchg ptr %buf, i64 %v monotonic, align 8: icv:1 ici:0
; ATOMIC: store i64 %v2, ptr %out, align 8: icv:1 ici:0

; CMPXCHG: %res = cmpxchg ptr %buf, i64 0, i64 %v monotonic monotonic, align 8: icv:1 ici:0
; CMPXCHG: store i64 %v2, ptr %out, align 8: icv:1 ici:0

; FULLINT: store i64 %v, ptr %buf, align 8: icv:1 ici:1
; FULLINT: store i64 %v2, ptr %out, align 8: icv:1 ici:1
