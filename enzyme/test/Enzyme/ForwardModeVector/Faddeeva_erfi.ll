; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -early-cse -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { { double, double }, { double, double }, { double, double } }

declare %struct.Gradients @__enzyme_fwddiff({ double, double } ({ double, double })*, ...)

declare { double, double } @Faddeeva_erfi({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfi({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define %struct.Gradients @test_derivative({ double, double } %x) {
entry:
  %0 = tail call %struct.Gradients ({ double, double } ({ double, double })*, ...) @__enzyme_fwddiff({ double, double } ({ double, double })* nonnull @tester,  metadata !"enzyme_width", i64 3, { double, double } %x, { double, double } { double 1.0, double 1.0 }, { double, double } { double 1.0, double 1.0 }, { double, double } { double 1.0, double 1.0 })
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x { double, double }] @fwddiffe3tester({ double, double } %in, [3 x { double, double }] %"in'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %[[a1:.+]] = extractvalue { double, double } %in, 1
; CHECK-NEXT:   %[[a3:.+]] = fmul fast double %[[a0]], %[[a0]]
; CHECK-NEXT:   %[[a2:.+]] = fmul fast double %[[a1]], %[[a1]]
; CHECK-NEXT:   %[[a4:.+]] = fsub fast double %[[a3]], %[[a2]]
; CHECK-NEXT:   %[[a5:.+]] = fmul fast double %[[a0]], %[[a1]]
; CHECK-NEXT:   %[[a6:.+]] = fadd fast double %[[a5]], %[[a5]]
; CHECK-NEXT:   %[[a7:.+]] = call fast double @llvm.exp.f64(double %[[a4]])
; CHECK-NEXT:   %[[a8:.+]] = call fast double @llvm.cos.f64(double %[[a6]])
; CHECK-NEXT:   %[[a9:.+]] = fmul fast double %[[a7]], %[[a8]]
; CHECK-NEXT:   %[[a10:.+]] = call fast double @llvm.sin.f64(double %[[a6]])
; CHECK-NEXT:   %[[a11:.+]] = fmul fast double %[[a7]], %[[a10]]
; CHECK-NEXT:   %[[a12:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a9]]
; CHECK-NEXT:   %[[a14:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a11]]
; CHECK-NEXT:   %[[a17:.+]] = extractvalue [3 x { double, double }] %"in'", 0, 0
; CHECK-NEXT:   %[[a29:.+]] = extractvalue [3 x { double, double }] %"in'", 1, 0
; CHECK-NEXT:   %[[a41:.+]] = extractvalue [3 x { double, double }] %"in'", 2, 0
; CHECK-NEXT:   %[[a18:.+]] = extractvalue [3 x { double, double }] %"in'", 0, 1
; CHECK-NEXT:   %[[a30:.+]] = extractvalue [3 x { double, double }] %"in'", 1, 1
; CHECK-NEXT:   %[[a42:.+]] = extractvalue [3 x { double, double }] %"in'", 2, 1

; CHECK-NEXT:   %[[a20:.+]] = fmul fast double %[[a17]], %[[a12]]
; CHECK-NEXT:   %[[a32:.+]] = fmul fast double %[[a29]], %[[a12]]
; CHECK-NEXT:   %[[a44:.+]] = fmul fast double %[[a41]], %[[a12]]

; CHECK-NEXT:   %[[a19:.+]] = fmul fast double %[[a18]], %[[a14]]
; CHECK-NEXT:   %[[a31:.+]] = fmul fast double %[[a30]], %[[a14]]
; CHECK-NEXT:   %[[a43:.+]] = fmul fast double %[[a42]], %[[a14]]

; CHECK-NEXT:   %[[a21:.+]] = fsub fast double %[[a20]], %[[a19]]
; CHECK-NEXT:   %[[a33:.+]] = fsub fast double %[[a32]], %[[a31]]
; CHECK-NEXT:   %[[a45:.+]] = fsub fast double %[[a44]], %[[a43]]

; CHECK-NEXT:   %[[a24:.+]] = fmul fast double %[[a17]], %[[a14]]
; CHECK-NEXT:   %[[a36:.+]] = fmul fast double %[[a29]], %[[a14]]
; CHECK-NEXT:   %[[a48:.+]] = fmul fast double %[[a41]], %[[a14]]

; CHECK-NEXT:   %[[a23:.+]] = fmul fast double %[[a12]], %[[a18]]
; CHECK-NEXT:   %[[a35:.+]] = fmul fast double %[[a12]], %[[a30]]
; CHECK-NEXT:   %[[a47:.+]] = fmul fast double %[[a12]], %[[a42]]

; CHECK-NEXT:   %[[a25:.+]] = fadd fast double %[[a24]], %[[a23]]
; CHECK-NEXT:   %[[a37:.+]] = fadd fast double %[[a36]], %[[a35]]


; CHECK-NEXT:   %[[a49:.+]] = fadd fast double %[[a48]], %[[a47]]


; CHECK-NEXT:   %[[r00:.+]] = insertvalue [3 x { double, double }] undef, double %[[a21]], 0, 0
; CHECK-NEXT:   %[[r01:.+]] = insertvalue [3 x { double, double }] %[[r00]], double %[[a25]], 0, 1

; CHECK-NEXT:   %[[r10:.+]] = insertvalue [3 x { double, double }] %[[r01]], double %[[a33]], 1, 0
; CHECK-NEXT:   %[[r11:.+]] = insertvalue [3 x { double, double }] %[[r10]], double %[[a37]], 1, 1

; CHECK-NEXT:   %[[r20:.+]] = insertvalue [3 x { double, double }] %[[r11]], double %[[a45]], 2, 0
; CHECK-NEXT:   %[[r21:.+]] = insertvalue [3 x { double, double }] %[[r20]], double %[[a49]], 2, 1

; CHECK-NEXT:   ret [3 x { double, double }] %[[r21]]
; CHECK-NEXT: }