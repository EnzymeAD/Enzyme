; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

define void @caller(float* %p, i1 %c) {
entry:
  %sel = select i1 %c, i64 0, i64 2
  %div = sdiv i64 4, %sel
  %gep = getelementptr inbounds float, float* %p, i64 %div
  store float 0.0, float* %gep, align 4
  %sel2 = select i1 %c, i64 -1, i64 2
  %div2 = sdiv i64 -9223372036854775808, %sel2
  %gep2 = getelementptr inbounds float, float* %p, i64 %div2
  store float 0.0, float* %gep2, align 4
  ret void
}

; CHECK: caller - {} |{[-1]:Pointer{{.*}}}:{} {[-1]:Integer}:{}
; CHECK:   %div = sdiv i64 4, %sel: {[-1]:Integer}
; CHECK:   %div2 = sdiv i64 -9223372036854775808, %sel2: {[-1]:Integer}
