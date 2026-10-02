; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; A sub-aggregate rebuilt by an insertvalue chain based on poison, whose last
; insert writes an untracked field. Walking back from the sret store reaches that
; insertvalue, and pushing its aggregate operand puts the bare poison on the
; worklist; the struct handling then looks for an extractvalue covering poison's
; tracked field, finds none, and used to fall through to an assert. A poison
; aggregate holds no live pointer, so there is nothing to root and the pass has
; no work to do here -- the tracked pointer is already in the returnRoots.

; CHECK-LABEL: define void @test_poison_insertvalue_base({{.*}} sret({{.*}}) %sret, {{.*}}"enzymejl_returnRoots"="1" %rroots
; CHECK: store {{.*}} %o, {{.*}} %sret
; CHECK: ret void

%tape = type { i8*, {} addrspace(10)*, i64 }
%outer = type { %tape }

define void @test_poison_insertvalue_base(%outer* sret(%outer) %sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %rroots, i8* %raw, {} addrspace(10)* %tracked, i64 %n) {
entry:
  %t0 = insertvalue %tape poison, i8* %raw, 0
  %t1 = insertvalue %tape %t0, {} addrspace(10)* %tracked, 1
  %t2 = insertvalue %tape %t1, i64 %n, 2
  %o = insertvalue %outer poison, %tape %t2, 0
  store %outer %o, %outer* %sret, align 8

  %g = getelementptr inbounds [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %rroots, i64 0, i64 0
  store {} addrspace(10)* %tracked, {} addrspace(10)** %g, align 8
  ret void
}
