; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | FileCheck %s --check-prefix=OPT; fi

declare double @__enzyme_fwddiff(...)
declare void @__enzyme_fwddiff_void(...)
declare <2 x double> @__enzyme_fwddiff_v2f64(...)
declare { double, i64 } @__enzyme_fwddiff_s(...)
declare [2 x double] @__enzyme_fwddiff_w2(...)

declare <2 x double> @llvm.masked.load.v2f64.p0(ptr, i32 immarg, <2 x i1>, <2 x double>)
declare void @llvm.masked.store.v2f64.p0(<2 x double>, ptr, i32 immarg, <2 x i1>)

define void @self_store(ptr %dst) {
entry:
  %v = load double, ptr %dst, align 8
  store double %v, ptr %dst, align 8
  ret void
}

define <2 x double> @masked_load_fn(ptr nocapture readonly %p, <2 x i1> %m, <2 x double> %pass) noinline {
entry:
  %v = call <2 x double> @llvm.masked.load.v2f64.p0(ptr %p, i32 8, <2 x i1> %m, <2 x double> %pass)
  ret <2 x double> %v
}

define void @masked_store_fn(ptr nocapture writeonly %p, <2 x i1> %m, <2 x double> %val) noinline {
entry:
  call void @llvm.masked.store.v2f64.p0(<2 x double> %val, ptr %p, i32 8, <2 x i1> %m)
  ret void
}

define { double, i64 } @load_double_i64(ptr nocapture readonly %src) noinline {
entry:
  %s = load { double, i64 }, ptr %src, align 8
  ret { double, i64 } %s
}

define { double, i64 } @load_packed_i128(ptr nocapture readonly "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}" %src) noinline {
entry:
  %raw = load i128, ptr %src, align 16
  %lo = trunc i128 %raw to i64
  %f = bitcast i64 %lo to double
  %hi128 = lshr i128 %raw, 64
  %hi = trunc i128 %hi128 to i64
  %s0 = insertvalue { double, i64 } undef, double %f, 0
  %s1 = insertvalue { double, i64 } %s0, i64 %hi, 1
  ret { double, i64 } %s1
}

define double @atomic_fadd(ptr %dst, double %x) noinline {
entry:
  %old = atomicrmw fadd ptr %dst, double %x monotonic, align 8
  ret double %old
}

define double @atomic_fmax(ptr %dst, double %x) noinline {
entry:
  %old = atomicrmw fmax ptr %dst, double %x monotonic, align 8
  ret double %old
}

define void @test_self_store(ptr %p, ptr %dp) {
entry:
  call void (...) @__enzyme_fwddiff_void(ptr @self_store, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp)
  ret void
}

define <2 x double> @test_masked_load(ptr %p, ptr %dp, <2 x i1> %m, <2 x double> %pass, <2 x double> %dpass) {
entry:
  %r = call <2 x double> (...) @__enzyme_fwddiff_v2f64(ptr @masked_load_fn, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_const", <2 x i1> %m, metadata !"enzyme_dup", <2 x double> %pass, <2 x double> %dpass)
  ret <2 x double> %r
}

define void @test_masked_store(ptr %p, ptr %dp, <2 x i1> %m, <2 x double> %v, <2 x double> %dv) {
entry:
  call void (...) @__enzyme_fwddiff_void(ptr @masked_store_fn, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_const", <2 x i1> %m, metadata !"enzyme_dup", <2 x double> %v, <2 x double> %dv)
  ret void
}

define { double, i64 } @test_load_double_i64(ptr %p, ptr %dp) {
entry:
  %r = call { double, i64 } (...) @__enzyme_fwddiff_s(ptr @load_double_i64, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp)
  ret { double, i64 } %r
}

define { double, i64 } @test_load_packed_i128(ptr %p, ptr %dp) {
entry:
  %r = call { double, i64 } (...) @__enzyme_fwddiff_s(ptr @load_packed_i128, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp)
  ret { double, i64 } %r
}

define [2 x double] @test_atomic_fadd_w2(ptr %p, ptr %dp0, ptr %dp1, double %x, double %dx0, double %dx1) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff_w2(ptr @atomic_fadd, metadata !"enzyme_runtime_activity", metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %p, ptr %dp0, ptr %dp1, metadata !"enzyme_dup", double %x, double %dx0, double %dx1)
  ret [2 x double] %r
}

define [2 x double] @test_atomic_fmax_w2(ptr %p, ptr %dp0, ptr %dp1, double %x, double %dx0, double %dx1) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff_w2(ptr @atomic_fmax, metadata !"enzyme_runtime_activity", metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %p, ptr %dp0, ptr %dp1, metadata !"enzyme_dup", double %x, double %dx0, double %dx1)
  ret [2 x double] %r
}

; CHECK-LABEL: define internal void @fwddiffeself_store(ptr nocapture %dst, ptr nocapture %"dst'")
; CHECK: entry:
; CHECK-NOT: !alias.scope
; CHECK-NOT: !noalias
; CHECK:   %"v'ipl" = load double, ptr %"dst'", align 8
; CHECK:   %[[CMP1:.+]] = icmp ne ptr %dst, %"dst'"
; CHECK:   %[[SEL:.+]] = select i1 %[[CMP1]], double %"v'ipl", double 0.000000e+00
; CHECK:   %v = load double, ptr %dst, align 8
; CHECK:   %[[CMP2:.+]] = icmp ne ptr %dst, %"dst'"
; CHECK:   br i1 %[[CMP2]], label %[[THEN:.+]], label %[[CONT:.+]]
; CHECK: [[THEN]]:
; CHECK:   store double %[[SEL]], ptr %"dst'", align 8
; CHECK-NOT: !alias.scope
; CHECK-NOT: !noalias
; CHECK:   br label %[[CONT]]
; CHECK: [[CONT]]:
; CHECK:   store double %v, ptr %dst, align 8
; CHECK-NOT: !alias.scope
; CHECK-NOT: !noalias
; CHECK:   ret void

; Under -O2, self_store must NOT clobber primal *%dst with 0.0 when %dst == %"dst'".
; OPT-LABEL: define void @test_self_store(ptr {{.*}}%p, ptr {{.*}}%dp)
; OPT-NOT: store
; OPT:   ret void

; CHECK-LABEL: define internal <2 x double> @fwddiffemasked_load_fn(ptr nocapture readonly %p, ptr nocapture %"p'", <2 x i1> %m, <2 x double> %pass, <2 x double> %"pass'")
; CHECK:   %[[LOAD:.+]] = call {{.*}}<2 x double> @llvm.masked.load.v2f64.p0(ptr %"p'", i32 8, <2 x i1> %m, <2 x double> %"pass'")
; CHECK:   %[[CMP:.+]] = icmp ne ptr %p, %"p'"
; CHECK:   %[[INACT:.+]] = select <2 x i1> %m, <2 x double> zeroinitializer, <2 x double> %[[LOAD]]
; CHECK:   %[[RES:.+]] = select i1 %[[CMP]], <2 x double> %[[LOAD]], <2 x double> %[[INACT]]
; CHECK:   ret <2 x double> %[[RES]]

; CHECK-LABEL: define internal void @fwddiffemasked_store_fn(ptr nocapture writeonly %p, ptr nocapture %"p'", <2 x i1> %m, <2 x double> %val, <2 x double> %"val'")
; CHECK:   %[[CMP:.+]] = icmp ne ptr %p, %"p'"
; CHECK:   br i1 %[[CMP]], label %[[THEN:.+]], label %[[CONT:.+]]
; CHECK: [[THEN]]:
; CHECK:   call void @llvm.masked.store.v2f64.p0(<2 x double> %"val'", ptr %"p'", i32 8, <2 x i1> %m)
; CHECK:   br label %[[CONT]]
; CHECK: [[CONT]]:
; CHECK:   call void @llvm.masked.store.v2f64.p0(<2 x double> %val, ptr %p, i32 8, <2 x i1> %m)
; CHECK:   ret void

; CHECK-LABEL: define internal { double, i64 } @fwddiffeload_double_i64(ptr nocapture readonly %src, ptr nocapture %"src'")
; CHECK:   %"s'ipl" = load { double, i64 }, ptr %"src'", align 8
; CHECK:   %[[CMP:.+]] = icmp ne ptr %src, %"src'"
; CHECK:   %[[F0:.+]] = extractvalue { double, i64 } %"s'ipl", 0
; CHECK:   %[[SEL:.+]] = select i1 %[[CMP]], double %[[F0]], double 0.000000e+00
; CHECK:   %[[INS:.+]] = insertvalue { double, i64 } %"s'ipl", double %[[SEL]], 0
; CHECK:   ret { double, i64 } %[[INS]]

; CHECK-LABEL: define internal { double, i64 } @fwddiffeload_packed_i128(ptr nocapture readonly {{.*}}%src, ptr nocapture {{.*}}%"src'")
; CHECK:   %"raw'ipl" = load i128, ptr %"src'", align 16
; CHECK:   %[[CMP:.+]] = icmp ne ptr %src, %"src'"
; CHECK:   %[[MASKED:.+]] = and i128 %"raw'ipl", -18446744073709551616
; CHECK:   %[[SEL:.+]] = select i1 %[[CMP]], i128 %"raw'ipl", i128 %[[MASKED]]
; CHECK:   trunc i128 %[[SEL]] to i64

; CHECK-LABEL: define internal [2 x double] @fwddiffe2atomic_fadd(ptr %dst, [2 x ptr] %"dst'", double %x, [2 x double] %"x'")
; CHECK:   %old = atomicrmw fadd ptr %dst, double %x monotonic, align 8
; CHECK:   %[[P0:.+]] = extractvalue [2 x ptr] %"dst'", 0
; CHECK:   %[[X0:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK:   %[[C0:.+]] = icmp ne ptr %dst, %[[P0]]
; CHECK:   br i1 %[[C0]], label %[[T0:.+]], label %[[M0:.+]]
; CHECK: [[T0]]:
; CHECK:   %[[A0:.+]] = atomicrmw fadd ptr %[[P0]], double %[[X0]] monotonic, align 8
; CHECK:   br label %[[M0]]
; CHECK: [[M0]]:
; CHECK:   %[[R0:.+]] = phi double [ %[[A0]], %[[T0]] ], [ 0.000000e+00, %entry ]
; CHECK:   %[[P1:.+]] = extractvalue [2 x ptr] %"dst'", 1
; CHECK:   %[[X1:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK:   %[[C1:.+]] = icmp ne ptr %dst, %[[P1]]
; CHECK:   br i1 %[[C1]], label %[[T1:.+]], label %[[M1:.+]]
; CHECK: [[T1]]:
; CHECK:   %[[A1:.+]] = atomicrmw fadd ptr %[[P1]], double %[[X1]] monotonic, align 8
; CHECK:   br label %[[M1]]
; CHECK: [[M1]]:
; CHECK:   %[[R1:.+]] = phi double [ %[[A1]], %[[T1]] ], [ 0.000000e+00, %[[M0]] ]
; CHECK:   ret [2 x double]

; CHECK-LABEL: define internal [2 x double] @fwddiffe2atomic_fmax(ptr %dst, [2 x ptr] %"dst'", double %x, [2 x double] %"x'")
; CHECK:   %[[P0:.+]] = extractvalue [2 x ptr] %"dst'", 0
; CHECK:   %[[X0:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK:   %[[C0:.+]] = icmp ne ptr %dst, %[[P0]]
; CHECK:   br i1 %[[C0]], label %[[T0:.+]], label %[[M0:.+]]
; CHECK: [[T0]]:
; CHECK:   %[[L0:.+]] = load atomic double, ptr %[[P0]] monotonic, align 8
; CHECK:   %[[S0:.+]] = select {{.*}}i1 %changed, double %[[X0]], double %[[L0]]
; CHECK:   store atomic double %[[S0]], ptr %[[P0]] monotonic, align 8
; CHECK:   br label %[[M0]]
; CHECK: [[M0]]:
; CHECK:   phi double [ %[[L0]], %[[T0]] ], [ 0.000000e+00, %{{.+}} ]
; CHECK:   %[[P1:.+]] = extractvalue [2 x ptr] %"dst'", 1
; CHECK:   %[[X1:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK:   %[[C1:.+]] = icmp ne ptr %dst, %[[P1]]
; CHECK:   br i1 %[[C1]], label %[[T1:.+]], label %[[M1:.+]]
; CHECK: [[T1]]:
; CHECK:   %[[L1:.+]] = load atomic double, ptr %[[P1]] monotonic, align 8
; CHECK:   %[[S1:.+]] = select {{.*}}i1 %changed, double %[[X1]], double %[[L1]]
; CHECK:   store atomic double %[[S1]], ptr %[[P1]] monotonic, align 8
; CHECK:   br label %[[M1]]
; CHECK: [[M1]]:
; CHECK:   phi double [ %[[L1]], %[[T1]] ], [ 0.000000e+00, %[[M0]] ]
; CHECK:   ret [2 x double]
