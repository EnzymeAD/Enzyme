; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg,adce)" -S | FileCheck %s

declare double @__enzyme_autodiff(double (double)*, ...)
declare noalias i8* @malloc(i64)
declare void @free(i8*)

define { double*, double } @augsquare(double %x) {
entry:
  %raw = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
  %m = bitcast i8* %raw to double*
  store double %x, double* %m, align 8
  %s0 = insertvalue { double*, double } undef, double* %m, 0
  %s1 = insertvalue { double*, double } %s0, double %x, 1
  ret { double*, double } %s1
}

; Only the active double field is used (exact Issue #597 case).
define double @square_float(double %x) {
entry:
  %ext = call { double*, double } @augsquare(double %x)
  %p = extractvalue { double*, double } %ext, 0
  %o = extractvalue { double*, double } %ext, 1
  %raw = bitcast double* %p to i8*
  call void @free(i8* %raw)
  %mul = fmul double %o, %o
  ret double %mul
}

; Both the duplicated pointer field and the active double field are used.
define double @square_both(double %x) {
entry:
  %ext = call { double*, double } @augsquare(double %x)
  %p = extractvalue { double*, double } %ext, 0
  %o = extractvalue { double*, double } %ext, 1
  %v = load double, double* %p, align 8
  %raw = bitcast double* %p to i8*
  call void @free(i8* %raw)
  %mul = fmul double %v, %o
  ret double %mul
}

; Only the duplicated pointer field is used (no differeturn needed).
define double @square_ptr(double %x) {
entry:
  %ext = call { double*, double } @augsquare(double %x)
  %p = extractvalue { double*, double } %ext, 0
  %v = load double, double* %p, align 8
  %raw = bitcast double* %p to i8*
  call void @free(i8* %raw)
  %mul = fmul double %v, %v
  ret double %mul
}

; Nested mixed struct { double*, { double*, double } }.
define { double*, { double*, double } } @nested_struct(double %x) {
entry:
  %inner = call { double*, double } @augsquare(double %x)
  %raw = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
  %m = bitcast i8* %raw to double*
  store double %x, double* %m, align 8
  %s0 = insertvalue { double*, { double*, double } } undef, double* %m, 0
  %s1 = insertvalue { double*, { double*, double } } %s0, { double*, double } %inner, 1
  ret { double*, { double*, double } } %s1
}

define double @caller_nested(double %x) {
entry:
  %r = call { double*, { double*, double } } @nested_struct(double %x)
  %p0 = extractvalue { double*, { double*, double } } %r, 0
  %inner = extractvalue { double*, { double*, double } } %r, 1
  %p1 = extractvalue { double*, double } %inner, 0
  %d = extractvalue { double*, double } %inner, 1
  %v0 = load double, double* %p0, align 8
  %v1 = load double, double* %p1, align 8
  %r0 = bitcast double* %p0 to i8*
  %r1 = bitcast double* %p1 to i8*
  call void @free(i8* %r0)
  call void @free(i8* %r1)
  %sum = fadd double %v0, %v1
  %tot = fadd double %sum, %d
  ret double %tot
}

define double @dsquare_float(double %x) {
entry:
  %call = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* @square_float, double %x)
  ret double %call
}

define double @dsquare_both(double %x) {
entry:
  %call = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* @square_both, double %x)
  ret double %call
}

define double @dsquare_ptr(double %x) {
entry:
  %call = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* @square_ptr, double %x)
  ret double %call
}

define double @dcaller_nested(double %x) {
entry:
  %call = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* @caller_nested, double %x)
  ret double %call
}

; CHECK: define internal { double } @diffesquare_float(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ext_augmented = call { { i8*, i8* }, { double*, double } } @augmented_augsquare(double %x)
; CHECK-NEXT:   %subcache = extractvalue { { i8*, i8* }, { double*, double } } %ext_augmented, 0
; CHECK-NEXT:   %ext = extractvalue { { i8*, i8* }, { double*, double } } %ext_augmented, 1
; CHECK-NEXT:   %o = extractvalue { double*, double } %ext, 1
; CHECK-NEXT:   %[[M0:.+]] = fmul fast double %differeturn, %o
; CHECK-NEXT:   %[[M1:.+]] = fmul fast double %differeturn, %o
; CHECK-NEXT:   %[[ADD:.+]] = fadd fast double %[[M0]], %[[M1]]
; CHECK-NEXT:   %[[INS:.+]] = insertvalue { double*, double } { double* null, double {{undef|poison}} }, double %[[ADD]], 1
; CHECK-NEXT:   %[[RES:.+]] = call { double } @diffeaugsquare(double %x, { double*, double } %[[INS]], { i8*, i8* } %subcache)
; CHECK-NEXT:   ret { double } %[[RES]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeaugsquare(double %x, { double*, double } %differeturn, { i8*, i8* } %tapeArg)
; CHECK:   %[[DRET:.+]] = extractvalue { double*, double } %differeturn, 1
; CHECK:   %[[LOAD:.+]] = load double, double* %"m'ipc"
; CHECK:   %[[SUM:.+]] = fadd fast double %[[DRET]], %[[LOAD]]
; CHECK:   %[[RET:.+]] = insertvalue { double } undef, double %[[SUM]], 0
; CHECK-NEXT:   ret { double } %[[RET]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesquare_both(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ext_augmented = call { { i8*, i8* }, { double*, double }, { double*, double } } @augmented_augsquare.1(double %x)
; CHECK-NEXT:   %subcache = extractvalue { { i8*, i8* }, { double*, double }, { double*, double } } %ext_augmented, 0
; CHECK-NEXT:   %ext = extractvalue { { i8*, i8* }, { double*, double }, { double*, double } } %ext_augmented, 1
; CHECK-NEXT:   %"ext'ac" = extractvalue { { i8*, i8* }, { double*, double }, { double*, double } } %ext_augmented, 2
; CHECK-NEXT:   %p = extractvalue { double*, double } %ext, 0
; CHECK-NEXT:   %"p'ipev" = extractvalue { double*, double } %"ext'ac", 0
; CHECK-NEXT:   %o = extractvalue { double*, double } %ext, 1
; CHECK-NEXT:   %v = load double, double* %p
; CHECK-NEXT:   %[[DO:.+]] = fmul fast double %differeturn, %o
; CHECK-NEXT:   %[[DV:.+]] = fmul fast double %differeturn, %v
; CHECK-NEXT:   %[[PL:.+]] = load double, double* %"p'ipev"
; CHECK-NEXT:   %[[PA:.+]] = fadd fast double %[[PL]], %[[DO]]
; CHECK-NEXT:   store double %[[PA]], double* %"p'ipev"
; CHECK-NEXT:   %[[INS2:.+]] = insertvalue { double*, double } { double* null, double {{undef|poison}} }, double %[[DV]], 1
; CHECK-NEXT:   %[[RES2:.+]] = call { double } @diffeaugsquare.2(double %x, { double*, double } %[[INS2]], { i8*, i8* } %subcache)
; CHECK-NEXT:   ret { double } %[[RES2]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeaugsquare.2(double %x, { double*, double } %differeturn, { i8*, i8* } %tapeArg)
; CHECK:   %[[DRET2:.+]] = extractvalue { double*, double } %differeturn, 1
; CHECK:   %[[LOAD2:.+]] = load double, double* %"m'ipc"
; CHECK:   %[[SUM2:.+]] = fadd fast double %[[DRET2]], %[[LOAD2]]
; CHECK:   %[[RET2:.+]] = insertvalue { double } undef, double %[[SUM2]], 0
; CHECK-NEXT:   ret { double } %[[RET2]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesquare_ptr(double %x, double %differeturn)
; CHECK:   %[[RES3:.+]] = call { double } @diffeaugsquare.4(double %x, { i8*, i8* } %subcache)
; CHECK-NEXT:   ret { double } %[[RES3]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeaugsquare.4(double %x, { i8*, i8* } %tapeArg)
; CHECK:   %[[LOAD3:.+]] = load double, double* %"m'ipc"
; CHECK:   %[[RET3:.+]] = insertvalue { double } undef, double %[[LOAD3]], 0
; CHECK-NEXT:   ret { double } %[[RET3]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffecaller_nested(double %x, double %differeturn)
; CHECK:   %[[NINS:.+]] = insertvalue { double*, { double*, double } } { double* null, { double*, double } { double* null, double {{undef|poison}} } }, double %differeturn, 1, 1
; CHECK-NEXT:   %[[NRES:.+]] = call { double } @diffenested_struct(double %x, { double*, { double*, double } } %[[NINS]], { i8*, i8*, { i8*, i8* }, { double*, double } } %subcache)
; CHECK-NEXT:   ret { double } %[[NRES]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffenested_struct(double %x, { double*, { double*, double } } %differeturn, { i8*, i8*, { i8*, i8* }, { double*, double } } %tapeArg)
; CHECK:   %[[ND:.+]] = extractvalue { double*, { double*, double } } %differeturn, 1, 1
; CHECK:   %[[NLOAD:.+]] = load double, double* %"m'ipc"
; CHECK:   %[[SINS:.+]] = insertvalue { double*, double } { double* null, double {{undef|poison}} }, double %[[ND]], 1
; CHECK-NEXT:   %[[SRES:.+]] = call { double } @diffeaugsquare.6(double %x, { double*, double } %[[SINS]], { i8*, i8* } %tapeArg1)
; CHECK-NEXT:   %[[SEXT:.+]] = extractvalue { double } %[[SRES]], 0
; CHECK-NEXT:   %[[NSUM:.+]] = fadd fast double %[[NLOAD]], %[[SEXT]]
; CHECK-NEXT:   %[[NSRET:.+]] = insertvalue { double } undef, double %[[NSUM]], 0
; CHECK-NEXT:   ret { double } %[[NSRET]]
; CHECK-NEXT: }
