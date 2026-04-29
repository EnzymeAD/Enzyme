; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

define void @caller({ { {} addrspace(10)* } }* %arg, { { {} addrspace(10)* } } addrspace(10)* %valid_ptr) {
; CHECK-LABEL: define void @caller({ { {} addrspace(10)* } }* %arg, { { {} addrspace(10)* } } addrspace(10)* %valid_ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { { { {} addrspace(10)* } }, [6 x i64] }, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { { { {} addrspace(10)* } }, [6 x i64] }, { { { {} addrspace(10)* } }, [6 x i64] }* %stack_sret, i32 0, i32 1
; CHECK-NEXT:   %1 = getelementptr inbounds { { { {} addrspace(10)* } }, [6 x i64] }, { { { {} addrspace(10)* } }, [6 x i64] }* %stack_sret, i32 0, i32 0
; CHECK-NEXT:   %stack_roots_AT = alloca [1 x {} addrspace(10)*], align 8
; CHECK-NEXT:   %arg_val = load { { {} addrspace(10)* } }, { { {} addrspace(10)* } }* %arg, align 8
; CHECK-NEXT:   store { { {} addrspace(10)* } } %arg_val, { { {} addrspace(10)* } }* %1, align 8
; CHECK-NEXT:   %gep1 = getelementptr inbounds { { {} addrspace(10)* } }, { { {} addrspace(10)* } }* %1, i32 0, i32 0
; CHECK-NEXT:   call void @callee({ { { {} addrspace(10)* } }, [6 x i64] }* sret({ { { {} addrspace(10)* } }, [6 x i64] }) %stack_sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %stack_roots_AT)
; CHECK-NEXT:   ret void
entry:
  %alloca = alloca { { {} addrspace(10)* } }
  %sret_box = alloca [6 x i64]
  %arg_val = load { { {} addrspace(10)* } }, { { {} addrspace(10)* } }* %arg
  store { { {} addrspace(10)* } } %arg_val, { { {} addrspace(10)* } }* %alloca
  %gep1 = getelementptr inbounds { { {} addrspace(10)* } }, { { {} addrspace(10)* } }* %alloca, i32 0, i32 0
  call void @callee({ { {} addrspace(10)* } }* "enzyme_sret"="test_type" "enzyme_type"="{[-1]:Pointer}" %alloca, [6 x i64]* "enzyme_sret"="test_type2" "enzyme_type"="{[-1]:Pointer}" %sret_box)
  ret void
}

define void @caller2(double* %arg) {
; CHECK-LABEL: define void @caller2(double* %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { double }, align 8
; CHECK-NEXT:   %val = load double, double* %arg, align 8
; CHECK-NEXT:   %gep = getelementptr inbounds { double }, { double }* %stack_sret, i32 0, i32 0
; CHECK-NEXT:   store double %val, double* %gep, align 8
; CHECK-NEXT:   call void @callee2({ double }* sret({ double }) %stack_sret)
; CHECK-NEXT:   %val2 = load double, double* %gep, align 8
; CHECK-NEXT:   store double %val2, double* %arg, align 8
; CHECK-NEXT:   ret void
entry:
  %sret_box = alloca { double }
  %val = load double, double* %arg, align 8
  %gep = getelementptr inbounds { double }, { double }* %sret_box, i32 0, i32 0
  store double %val, double* %gep, align 8
  call void @callee2({ double }* "enzyme_sret"="test_type3" "enzyme_type"="{[-1]:Pointer}" %sret_box)
  %val2 = load double, double* %gep, align 8
  store double %val2, double* %arg, align 8
  ret void
}

define void @callee({ { {} addrspace(10)* } }* "enzyme_sret"="test_type" "enzyme_type"="{[-1]:Pointer}" %sret_return, [6 x i64]* "enzyme_sret"="test_type2" "enzyme_type"="{[-1]:Pointer}" %sret_return_prime) {
; CHECK-LABEL: define void @callee({ { { {} addrspace(10)* } }, [6 x i64] }* noalias sret({ { { {} addrspace(10)* } }, [6 x i64] }) %0, [1 x {} addrspace(10)*]* noalias writeonly "enzymejl_returnRoots"="1" %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = getelementptr inbounds { { { {} addrspace(10)* } }, [6 x i64] }, { { { {} addrspace(10)* } }, [6 x i64] }* %0, i32 0, i32 1
; CHECK-NEXT:   %3 = getelementptr inbounds { { { {} addrspace(10)* } }, [6 x i64] }, { { { {} addrspace(10)* } }, [6 x i64] }* %0, i32 0, i32 0
; CHECK-NEXT:   %val = load [6 x i64], [6 x i64]* %2, align 8
; CHECK-NEXT:   store [6 x i64] %val, [6 x i64]* %2, align 8
; CHECK-NEXT:   %4 = getelementptr inbounds [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %1, i64 0, i32 0
; CHECK-NEXT:   %5 = getelementptr inbounds { { {} addrspace(10)* } }, { { {} addrspace(10)* } }* %3, i64 0, i32 0, i32 0
; CHECK-NEXT:   %6 = load {} addrspace(10)*, {} addrspace(10)** %5, align 8
; CHECK-NEXT:   store {} addrspace(10)* %6, {} addrspace(10)** %4, align 8
; CHECK-NEXT:   ret void
entry:
  %val = load [6 x i64], [6 x i64]* %sret_return_prime, align 8
  store [6 x i64] %val, [6 x i64]* %sret_return_prime, align 8
  ret void
}

define void @callee2({ double }* "enzyme_sret"="test_type3" "enzyme_type"="{[-1]:Pointer}" %sret_return) {
; CHECK-LABEL: define void @callee2({ double }* noalias sret({ double }) %0)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep = getelementptr inbounds { double }, { double }* %0, i32 0, i32 0
; CHECK-NEXT:   %val = load double, double* %gep, align 8
; CHECK-NEXT:   store double %val, double* %gep, align 8
; CHECK-NEXT:   ret void
entry:
  %gep = getelementptr inbounds { double }, { double }* %sret_return, i32 0, i32 0
  %val = load double, double* %gep, align 8
  store double %val, double* %gep, align 8
  ret void
}
