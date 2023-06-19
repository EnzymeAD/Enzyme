; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -early-cse -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { { double, double }, { double, double }, { double, double } }

declare %struct.Gradients @__enzyme_fwddiff({ double, double } ({ double, double })*, ...)

declare { double, double } @Faddeeva_erfc({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfc({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define %struct.Gradients @test_derivative({ double, double } %x) {
entry:
  %0 = tail call %struct.Gradients ({ double, double } ({ double, double })*, ...) @__enzyme_fwddiff({ double, double } ({ double, double })* @tester,  metadata !"enzyme_width", i64 3, { double, double } %x, { double, double } { double 1.0, double 1.0 }, { double, double } { double 1.0, double 1.0 }, { double, double } { double 1.0, double 1.0 })
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x { double, double }] @fwddiffe3tester({ double, double } %in, [3 x { double, double }] %"in'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %[[a1:.+]] = extractvalue { double, double } %in, 1
; CHECK-DAG:    %[[a2:.+]] = fmul fast double %[[a1]], %[[a1]]
; CHECK-DAG:    %[[a3:.+]] = fmul fast double %[[a0]], %[[a0]]
; CHECK-NEXT:   %[[a4:.+]] = fsub fast double %[[a3]], %[[a2]]
; CHECK-NEXT:   %[[a5:.+]] = fmul fast double %[[a0]], %[[a1]]
; CHECK-NEXT:   %[[a6:.+]] = fadd fast double %[[a5]], %[[a5]]
; CHECK-NEXT:   %[[a7:.+]] = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %[[a4]]
; CHECK-NEXT:   %[[a8:.+]] = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %[[a6]]
; CHECK-NEXT:   %[[a9:.+]] = call fast double @llvm.exp.f64(double %[[a7]])
; CHECK-NEXT:   %[[a10:.+]] = call fast double @llvm.cos.f64(double %[[a8]])
; CHECK-NEXT:   %[[a11:.+]] = fmul fast double %[[a9]], %[[a10]]
; CHECK-NEXT:   %[[a12:.+]] = call fast double @llvm.sin.f64(double %[[a8]])
; CHECK-NEXT:   %[[a13:.+]] = fmul fast double %[[a9]], %[[a12]]
; CHECK-NEXT:   %[[a14:.+]] = fmul fast double 0xBFF20DD750429B6D, %[[a11]]
; CHECK-NEXT:   %[[a16:.+]] = fmul fast double 0xBFF20DD750429B6D, %[[a13]]
; CHECK-NEXT:   %[[a19:.+]] = extractvalue [3 x { double, double }] %"in'", 0, 0
; CHECK-NEXT:   %[[a31:.+]] = extractvalue [3 x { double, double }] %"in'", 1, 0
; CHECK-NEXT:   %[[a43:.+]] = extractvalue [3 x { double, double }] %"in'", 2, 0
; CHECK-NEXT:   %[[a20:.+]] = extractvalue [3 x { double, double }] %"in'", 0, 1
; CHECK-NEXT:   %[[a32:.+]] = extractvalue [3 x { double, double }] %"in'", 1, 1
; CHECK-NEXT:   %[[a44:.+]] = extractvalue [3 x { double, double }] %"in'", 2, 1

; CHECK-NEXT:    %[[a22:.+]] = fmul fast double %[[a19]], %[[a14]]
; CHECK-NEXT:    %[[a34:.+]] = fmul fast double %[[a31]], %[[a14]]
; CHECK-NEXT:    %[[a46:.+]] = fmul fast double %[[a43]], %[[a14]]


; CHECK-NEXT:    %[[a21:.+]] = fmul fast double %[[a20]], %[[a16]]
; CHECK-NEXT:    %[[a33:.+]] = fmul fast double %[[a32]], %[[a16]]
; CHECK-NEXT:    %[[a45:.+]] = fmul fast double %[[a44]], %[[a16]]


; CHECK-NEXT:   %[[a23:.+]] = fsub fast double %[[a22]], %[[a21]]
; CHECK-NEXT:   %[[a35:.+]] = fsub fast double %[[a34]], %[[a33]]
; CHECK-NEXT:   %[[a47:.+]] = fsub fast double %[[a46]], %[[a45]]


; CHECK-NEXT:    %[[a26:.+]] = fmul fast double %[[a19]], %[[a16]]
; CHECK-NEXT:    %[[a38:.+]] = fmul fast double %[[a31]], %[[a16]]
; CHECK-NEXT:    %[[a50:.+]] = fmul fast double %[[a43]], %[[a16]]

; CHECK-NEXT:    %[[a25:.+]] = fmul fast double %[[a14]], %[[a20]]
; CHECK-NEXT:    %[[a37:.+]] = fmul fast double %[[a14]], %[[a32]]
; CHECK-NEXT:    %[[a49:.+]] = fmul fast double %[[a14]], %[[a44]]

; CHECK-NEXT:   %[[a27:.+]] = fadd fast double %[[a26]], %[[a25]]
; CHECK-NEXT:   %[[a39:.+]] = fadd fast double %[[a38]], %[[a37]]
; CHECK-NEXT:   %[[a51:.+]] = fadd fast double %[[a50]], %[[a49]]

; CHECK-NEXT:   %[[r00:.+]] = insertvalue [3 x { double, double }] undef, double %[[a23]], 0, 0
; CHECK-NEXT:   %[[r01:.+]] = insertvalue [3 x { double, double }] %[[r00]], double %[[a27]], 0, 1

; CHECK-NEXT:   %[[r10:.+]] = insertvalue [3 x { double, double }] %[[r01]], double %[[a35]], 1, 0
; CHECK-NEXT:   %[[r11:.+]] = insertvalue [3 x { double, double }] %[[r10]], double %[[a39]], 1, 1

; CHECK-NEXT:   %[[r20:.+]] = insertvalue [3 x { double, double }] %[[r11]], double %[[a47]], 2, 0
; CHECK-NEXT:   %[[r21:.+]] = insertvalue [3 x { double, double }] %[[r20]], double %[[a51]], 2, 1

; CHECK-NEXT:   ret [3 x { double, double }] %[[r21]]
; CHECK-NEXT: }
