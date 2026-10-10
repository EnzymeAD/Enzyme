; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s 2>&1 | FileCheck %s

; The tracked pointer of an aggregate returned by a call is stored into the sret
; through a chain of extractvalues, `extractvalue (extractvalue %a, 0), 0`, while
; the store into the returnRoots extracts the same field with a single
; multi-index extractvalue, `extractvalue %a, 0, 0`. Both are the same value, so
; the pointer is rooted. Matching the two only by SSA value used to miss this,
; report "Could not find use of stored value" and reroot the sret.

; CHECK-NOT: Could not find use of stored value
; CHECK-LABEL: define void @test_extractvalue_chain({{.*}} sret({{.*}}) %sret, {{.*}}"enzymejl_returnRoots"="1" %rroots
; CHECK: store {{.*}} %p, {{.*}} %sp
; CHECK: store {{.*}} %r, {{.*}} %g
; CHECK-NEXT: ret void

%inner = type { {} addrspace(10)*, i64 }
%agg = type { %inner, double }

declare %agg @make({} addrspace(10)*)

define void @test_extractvalue_chain(%inner* sret(%inner) %sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %rroots, {} addrspace(10)* %x) {
entry:
  %a = call %agg @make({} addrspace(10)* %x)
  %sub = extractvalue %agg %a, 0
  %p = extractvalue %inner %sub, 0
  %n = extractvalue %inner %sub, 1
  %sp = getelementptr inbounds %inner, %inner* %sret, i64 0, i32 0
  store {} addrspace(10)* %p, {} addrspace(10)** %sp, align 8
  %sn = getelementptr inbounds %inner, %inner* %sret, i64 0, i32 1
  store i64 %n, i64* %sn, align 8

  %r = extractvalue %agg %a, 0, 0
  %g = getelementptr inbounds [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %rroots, i64 0, i64 0
  store {} addrspace(10)* %r, {} addrspace(10)** %g, align 8
  ret void
}
