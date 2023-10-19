; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=_Z8simulatef -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=_Z8simulatef -S | FileCheck %s

define float @_ZNK11SystemModel1fERKf(i8* %p, float* %u) {
entry:
  %0 = load float, float* %u, align 4
  ret float %0
}

; Function Attrs: uwtable
define float @_Z8simulatef(float %input){
entry:
  %u.addr.i = alloca float, align 4
  %sys = alloca float (i8*, float*)*, align 8
  store float (i8*, float*)* @_ZNK11SystemModel1fERKf, float (i8*, float*)** %sys, align 8
  store float %input, float* %u.addr.i, align 4
  %0 = load float (i8*, float*)*, float (i8*, float*)** %sys, align 8
  %p = bitcast float (i8*, float*)** %sys to i8*
  %call.i = call fast float %0(i8* %p, float* %u.addr.i)
  ret float %call.i
}

; CHECK: float %input: icv:0
; CHECK: entry
; CHECK-NEXT:   %u.addr.i = alloca float, align 4: icv:0 ici:1
; CHECK-NEXT:   %sys = alloca float (i8*, float*)*, align 8: icv:0 ici:1
; CHECK-NEXT:   store float (i8*, float*)* @_ZNK11SystemModel1fERKf, float (i8*, float*)** %sys, align 8: icv:1 ici:0
; CHECK-NEXT:   store float %input, float* %u.addr.i, align 4: icv:1 ici:0
; CHECK-NEXT:   %0 = load float (i8*, float*)*, float (i8*, float*)** %sys, align 8: icv:0 ici:1
; CHECK-NEXT:   %p = bitcast float (i8*, float*)** %sys to i8*: icv:0 ici:1
; CHECK-NEXT:   %call.i = call fast float %0(i8* %p, float* %u.addr.i): icv:0 ici:0
; CHECK-NEXT:   ret float %call.i: icv:1 ici:1