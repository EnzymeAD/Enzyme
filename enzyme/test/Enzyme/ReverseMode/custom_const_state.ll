; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false  -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Verifies the combined reverse-mode custom-rule path
; (!enzyme_augment/!enzyme_gradient): a Const `ptr %state` argument the
; custom rule reads and writes, but the primal never touches, reaches the
; generated augment/gradient call as a real pointer, and the generated
; derivative definition/call site don't keep a readnone/readonly/writeonly
; claim that no longer holds once the custom rule's memory access counts.
;
; %state carries readnone/readonly/writeonly as a *parameter* attribute, on
; both @outer_*'s definition and the argument at its call to @atom -- not as
; a whole-call function attribute. The fix reads parameter attributes off
; the function being cloned (FunctionUtils.cpp) and off the specific call
; (AdjointGenerator.h); asserting them directly exercises both copy sites
; without depending on some earlier pass inferring one from the other.
;
; @atom is a pure `declare`, so asserting these on a call to it doesn't
; contradict any body -- unlike a defined callee (see
; ForwardMode/callee_const_arg.ll, which covers that case). The three
; variants (_rn/_ro/_wo) each stimulate one of the three attributes the fix
; removes. `nocapture` is asserted alongside each and must survive on the
; generated derivative, since the fix never touches it.
;
; Does not exercise tape encoding/data flow -- the fix only touches which
; ReadNone/ReadOnly/WriteOnly attributes survive on a Const argument, not
; how augment's tape reaches gradient, so an empty `{}` tape is sufficient.
; Does not cover the split augment/reverse API. See EnzymeAD/Enzyme.jl#3570.

define internal { {}, double } @augment_atom(double %x, ptr nocapture %state, ptr nocapture %dstate) {
entry:
  %old = load double, ptr %state
  %new = fadd fast double %old, 1.000000e+00
  store double %new, ptr %state
  %struct1 = insertvalue { {}, double } undef, double %new, 1
  ret { {}, double } %struct1
}

define internal { double } @gradient_atom(double %x, ptr nocapture %state, ptr nocapture %dstate, double %differet, {} %tapeArg) {
entry:
  %struct1 = insertvalue { double } undef, double %differet, 0
  ret { double } %struct1
}

declare !enzyme_augment !{ptr @augment_atom} !enzyme_gradient !{ptr @gradient_atom} double @atom(double %x, ptr %state)

define dso_local double @outer_rn(double %x, ptr nocapture readnone %state) {
entry:
  %call = tail call fast double @atom(double %x, ptr nocapture readnone %state)
  ret double %call
}

define dso_local double @outer_ro(double %x, ptr nocapture readonly %state) {
entry:
  %call = tail call fast double @atom(double %x, ptr nocapture readonly %state)
  ret double %call
}

define dso_local double @outer_wo(double %x, ptr nocapture writeonly %state) {
entry:
  %call = tail call fast double @atom(double %x, ptr nocapture writeonly %state)
  ret double %call
}

define dso_local double @douter_rn(double %x, ptr %state) {
entry:
  %0 = tail call double (ptr, ...) @__enzyme_autodiff(ptr nonnull @outer_rn, double %x, metadata !"enzyme_const", ptr %state)
  ret double %0
}

define dso_local double @douter_ro(double %x, ptr %state) {
entry:
  %0 = tail call double (ptr, ...) @__enzyme_autodiff(ptr nonnull @outer_ro, double %x, metadata !"enzyme_const", ptr %state)
  ret double %0
}

define dso_local double @douter_wo(double %x, ptr %state) {
entry:
  %0 = tail call double (ptr, ...) @__enzyme_autodiff(ptr nonnull @outer_wo, double %x, metadata !"enzyme_const", ptr %state)
  ret double %0
}

declare double @__enzyme_autodiff(ptr, ...)

; CHECK directives follow the order these definitions appear in the
; generated module (FileCheck scans forward only). Generated call lines end
; in `{{$}}` so a reintroduced readnone/readonly/writeonly on %state can't
; slip in at the end either (LLVM always prints it inline before the
; parameter, never after).
;
; augment_atom/gradient_atom's own `nocapture` on %state/%dstate is real
; (neither stores the pointer) and unaffected by the fix -- confirm it
; survives, unlike ReadNone/ReadOnly/WriteOnly.
; CHECK: define internal { {}, double } @augment_atom(double %x, ptr nocapture %state, ptr nocapture{{.*}} %dstate)
; CHECK: define internal { double } @gradient_atom(double %x, ptr nocapture{{.*}} %state, ptr nocapture{{.*}} %dstate, double %differet, {} %tapeArg)

; CHECK: define internal { double } @diffeouter_rn(double %x, ptr nocapture %state, double %differeturn) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @fixgradient_atom(double %x, ptr nocapture %state, double %differeturn){{$}}
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }

; CHECK: define internal { double } @fixgradient_atom(double %arg0, ptr %arg1, double %arg2) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { {}, double } @fixaugment_atom(double %arg0, ptr %arg1)
; CHECK-NEXT:   %1 = call { double } @fixgradient_atom.1(double %arg0, ptr %arg1, double %arg2, {} undef)
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal { {}, double } @fixaugment_atom(double %arg0, ptr %arg1) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { {}, double } @augment_atom(double %arg0, ptr %arg1, ptr %arg1)
; CHECK-NEXT:   ret { {}, double } %0
; CHECK-NEXT: }

; CHECK: define internal { double } @fixgradient_atom.1(double %arg0, ptr %arg1, double %postarg0, {} %postarg1) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @gradient_atom(double %arg0, ptr %arg1, ptr %arg1, double %postarg0, {} %postarg1)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeouter_ro(double %x, ptr nocapture %state, double %differeturn) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @fixgradient_atom(double %x, ptr nocapture %state, double %differeturn){{$}}
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeouter_wo(double %x, ptr nocapture %state, double %differeturn) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @fixgradient_atom(double %x, ptr nocapture %state, double %differeturn){{$}}
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }
