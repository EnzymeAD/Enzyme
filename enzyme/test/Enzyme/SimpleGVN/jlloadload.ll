; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

declare  void @julia___conv_filter__271_37475({ { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }* noalias nocapture nofree noundef nonnull writeonly sret({ { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }) align 8 dereferenceable(72) %0);

; Function Attrs: noinline
define private {} addrspace(10)* @julia__conv_filter__37469([1 x {} addrspace(10)*]* %return_roots) {
top:
  %0 = alloca { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }, align 8
  call fastcc void @julia___conv_filter__271_37475({ { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }* noalias nocapture nofree noundef nonnull writeonly sret({ { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }) align 8 dereferenceable(72) %0)
  %a5 = getelementptr inbounds { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }, { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }* %0, i64 0, i32 0, i32 0
  %a6 = load {} addrspace(10)*, {} addrspace(10)** %a5, align 8
  %a7 = getelementptr inbounds [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %return_roots, i64 0, i64 0
  store {} addrspace(10)* %a6, {} addrspace(10)** %a7, align 8
  %srcloccs2 = getelementptr inbounds { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }, { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }* %0, i64 0, i32 0, i32 0
  %a8 = load {} addrspace(10)*, {} addrspace(10)** %srcloccs2, align 8
  ret {} addrspace(10)* %a8
}

; CHECK: define private {} addrspace(10)* @julia__conv_filter__37469([1 x {} addrspace(10)*]* %return_roots)
; CHECK-NEXT: top:
; CHECK-NEXT:   %0 = alloca { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }, align 8
; CHECK-NEXT:   call fastcc void @julia___conv_filter__271_37475({ { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }* noalias nocapture nofree noundef nonnull writeonly sret({ { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }) align 8 dereferenceable(72) %0)
; CHECK-NEXT:   %a5 = getelementptr inbounds { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }, { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }* %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %a6 = load {} addrspace(10)*, {} addrspace(10)** %a5, align 8
; CHECK-NEXT:   %a7 = getelementptr inbounds [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %return_roots, i64 0, i64 0
; CHECK-NEXT:   store {} addrspace(10)* %a6, {} addrspace(10)** %a7, align 8
; CHECK-NEXT:   %srcloccs2 = getelementptr inbounds { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }, { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [4 x i64] }* %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   ret {} addrspace(10)* %a6
; CHECK-NEXT: }

