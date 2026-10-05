; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s 2>&1 | FileCheck %s

; Without a frontend to register a CustomErrorHandler, being unable to tell
; whether an sret needs rerooting is reported as a warning, and the argument is
; conservatively rerooted.

; CHECK: warning: {{.*}} in function callee {{.*}} Enzyme: Unknown user of sret-like argument

declare void @unknown({ { {} addrspace(10)* } }*)

; CHECK: define void @callee({{.*}} noalias sret({ { { {{.*}} } }, [1 x {{.*}}] }) %0)
; CHECK: call void @unknown(
define void @callee({ { {} addrspace(10)* } }* "enzyme_sret"="test_type" %sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %roots) {
entry:
  call void @unknown({ { {} addrspace(10)* } }* %sret)
  ret void
}
