; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; A double split into two i32 values (as SROA does on 32-bit targets), then reassembled.
define double @tester(double %x, ptr %out_lo, ptr %out_hi) {
entry:
  %bc = bitcast double %x to i64
  %lo = trunc i64 %bc to i32
  %shr = lshr i64 %bc, 32
  %hi = trunc i64 %shr to i32
  store i32 %lo, ptr %out_lo, align 4
  store i32 %hi, ptr %out_hi, align 4
  %lo_ext = zext i32 %lo to i64
  %hi_ext = zext i32 %hi to i64
  %hi_shl = shl nuw i64 %hi_ext, 32
  %repack = or i64 %hi_shl, %lo_ext
  %res = bitcast i64 %repack to double
  %m = fmul double %res, %res
  ret double %m
}

define double @test_derivative(double %x, ptr %out_lo, ptr %dout_lo, ptr %out_hi, ptr %dout_hi) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(ptr nonnull @tester, double %x, double 1.0, ptr %out_lo, ptr %dout_lo, ptr %out_hi, ptr %dout_hi)
  ret double %0
}

declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'", ptr {{.*}}%out_lo, ptr {{.*}}%"out_lo'", ptr {{.*}}%out_hi, ptr {{.*}}%"out_hi'")
; CHECK:   %[[dbc:.+]] = bitcast double %"x'" to i64
; CHECK:   %[[dlo:.+]] = trunc i64 %[[dbc]] to i32
; CHECK:   %[[dshr:.+]] = lshr i64 %[[dbc]], 32
; CHECK:   %[[dhi:.+]] = trunc i64 %[[dshr]] to i32
; CHECK:   store i32 %[[dlo]], ptr %"out_lo'", align 4
; CHECK:   store i32 %[[dhi]], ptr %"out_hi'", align 4
; CHECK:   %[[dlo_ext:.+]] = zext i32 %[[dlo]] to i64
; CHECK:   %[[dhi_ext:.+]] = zext i32 %[[dhi]] to i64
; CHECK:   %[[dhi_shl:.+]] = shl i64 %[[dhi_ext]], 32
; CHECK:   %[[drepack:.+]] = or i64 %[[dhi_shl]], %[[dlo_ext]]
; CHECK:   %[[dres:.+]] = bitcast i64 %[[drepack]] to double
