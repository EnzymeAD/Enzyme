; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-julia-addr-load=1 -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-julia-addr-load=1 -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"

%Inner = type { double, ptr addrspace(10) }
%Mixed = type { i64, [2 x %Inner], ptr addrspace(11) }
%AllJl = type { ptr addrspace(10), [2 x ptr addrspace(13)] }

declare %Mixed @make_mixed()
declare %AllJl @make_alljl()

define double @callee(double %x, i64 %idx) {
entry:
  %m0 = call %Mixed @make_mixed()
  %m1 = insertvalue %Mixed %m0, i64 %idx, 0
  %m2 = insertvalue %Mixed %m1, double %x, 1, 0, 0
  %aj = call %AllJl @make_alljl()
  %y = extractvalue %Mixed %m2, 1, 0, 0
  ret double %y
}

; CHECK: entry
; CHECK-NEXT:   %m0 = call %Mixed @make_mixed(): {[16]:Pointer, [32]:Pointer, [40]:Pointer}
; CHECK-NEXT:   %m1 = insertvalue %Mixed %m0, i64 %idx, 0: {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Integer, [5]:Integer, [6]:Integer, [7]:Integer, [16]:Pointer, [32]:Pointer, [40]:Pointer}
; CHECK-NEXT:   %m2 = insertvalue %Mixed %m1, double %x, 1, 0, 0: {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Integer, [5]:Integer, [6]:Integer, [7]:Integer, [8]:Float@double, [16]:Pointer, [32]:Pointer, [40]:Pointer}
; CHECK-NEXT:   %aj = call %AllJl @make_alljl(): {[-1]:Pointer}
; CHECK-NEXT:   %y = extractvalue %Mixed %m2, 1, 0, 0: {[-1]:Float@double}
; CHECK-NEXT:   ret double %y: {}
