/*
* Design:
*/

class BLASType<int _n, bit _active> { 
  int nelem = _n; 
  bit active = _active;
}

// scalar types
def len   : BLASType<1, 0>; // num of elements
def hbw   : BLASType<1, 0>; // half matrix-bandwith}
def bw    : BLASType<1, 0>; // matrix-bandwith}
def fp    : BLASType<1, 1>;  // floating point

// packed array of size ( n * (n  + 1) / 2 )
class ap<list<string> _args> : BLASType<1, 1> {
  list<string> args = _args;
}
class vinc<list<string> _args> : BLASType<2, 1> {
  list<string> args = _args;
}
class mld<list<string> _args> : BLASType<2, 1> {
  list<string> args = _args;
}

class blas_modes<list<string> _modes> : BLASType<1, 0> {
  list<string> modes = _modes;
}

// cblas_layout must(!) allways be called layout, or the world will burn
def cblas_layout  : blas_modes<["CblasRowMajor","CblasColMajor"]>;
def uplo          : blas_modes<["U","L","G"]> ;
def cblas_uplo    : blas_modes<["CblasUpper","CblasLower"]> ;
def trans         : blas_modes<["N","T","C"]> ;
def cblas_trans   : blas_modes<["CblasNoTrans","CblasTrans","CblasConjTrans"]> ;
def diag          : blas_modes<["U","N"]> ;
def cblas_diag    : blas_modes<["CblasUnit","CblasNonUnit"]> ;
def side          : blas_modes<["L", "R"]> ;
def cblas_side    : blas_modes<["CblasLeft","CblasRight"]>;


class CallBlasPattern<dag patternToMatch, list<string> mutables, list<BLASType> paramTypes, 
                      list<dag> resultOps> {
  dag PatternToMatch = patternToMatch;
  list<BLASType> inputTypes = paramTypes;
  list<dag> ArgDerivatives = resultOps;
  list<string> mutable = mutables;
}


class MagicInst : Inst<"blas">;
def Rows : MagicInst; // given a transpose, normal rows, normal cols get the true rows, aka normal rows if N else normal cols
def Concat : MagicInst;

// if !cache_A, then just use $lda.
// if cache_A, then check $transa.
// if $transa = 'N', then use $m.
// else use $n
def ld : MagicInst; // (ld $A, $transa $lda, $m, $n): 

// placeholder name
class BlasCall<string _s> {
  string s = _s;
}

// only applicable to arguments being modified
// will cache the input values to be used later
class input<string _name> {
  string name = _name;
}

class Constant<string _value> {
  string value = _value;
}
class Char<string _value> {
  string value = _value;
}
class Alloca<int _value> {
  int value = _value;
}

class transpose<string _name> {
  string name = _name;
}

class Seq<list<string> _args> {
  list<string> args = _args;
}

class use<string _name> {
  string name = _name;
}

class FrobInnerProd<string _tmp> {
  string unused = _tmp;
}

class DiagUpdateSPMV<string _tmp> {
  string unused = _tmp;
}

// General note: If return is scalar, return it. If return is vec, update it.

// Lv 1 \wo rotations
def scal : CallBlasPattern<(Op $n, $alpha, $x, $incx),
                  ["x"],[len, fp, vinc<["n"]>],
                  [
                  // dot must proceed scal, because scal modifies (Shadow $x)
                  (BlasCall<"dot"> $n, $x, (Shadow $x)),
                  (BlasCall<"scal"> $n, $alpha, (Shadow $x))
                  ]
                  >;

// def lacpy : CallBlasPattern<(Op $layout, $m, $n, $A, $lda, $B, $ldb),
//                   ["B"],[cblas_layout, len, len, mld<["m", "n"]>, mld<["m", "n"]>],
//                   [
//                   (AssertingInactiveArg), // from
//                   (AssertingInactiveArg), // to
//                   ]
//                   >;

def lascl : CallBlasPattern<(Op $layout, $type, $kl, $ku, $cfrom, $cto, $m, $n, $A, $lda, $info),
                  ["A"],[cblas_layout, uplo, len, len, fp, fp, len, len, mld<["m", "n"]>, len],
                  [
                  (AssertingInactiveArg), // cfrom
                  (AssertingInactiveArg), // cto
                  (AssertingInactiveArg)  // A
                  ]
                  >;

def axpy : CallBlasPattern<(Op $n, $alpha, $x, $incx, $y, $incy),
                  ["y"],[len, fp, vinc<["n"]>, vinc<["n"]>],
                  [
                  (BlasCall<"dot"> $n, (Shadow $y), $x),
                  (BlasCall<"axpy"> $n, $alpha, (Shadow $y), (Shadow $x)),
                  (InactiveArg) // y = alpha*x + y, so nothing to do here
                  ]
                  >;

def dot : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
                  [],[len, vinc<["n"]>, vinc<["n"]>],
                  [
                  (BlasCall<"axpy"> $n, DiffeRet, $y, (Shadow $x)),
                  (BlasCall<"axpy"> $n, DiffeRet, $x, (Shadow $y)),
                  ]
                  >;

// def nrm2 : CallBlasPattern<(Op $n, $x, $incx),
//                   [],[len, vinc],
//                   [(FDiv (BlasCall<"scal"> $n, DiffeRet, $x, $incx), Ret<"">)]
//                   >;


def copy : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
                  ["y"],[len, vinc<["n"]>, vinc<["n"]>],
                  [
                  (InactiveArg),// copy moves x into y, so x is never modified.
                  (BlasCall<"axpy"> $n, Constant<"1.0">, (Shadow $y), (Shadow $x))
                  ]
                  >;

// def swap : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
//                   ["x","y"],[len, vinc, vinc],
//                   [
//                   (BlasCall<"swap"> $n, (Shadow $x), $incx, (Shadow $y), $incy),
//                   (InactiveArg),// one swap is enough, will only be used iff x and y active anyway
//                   ]
//                   >;

// def asum : CallBlasPattern<(Op $n, $x, $incx),
//                   [],[len, vinc],
//                   [(BlasCall<"axpy"> DiffeRet, sign(<input<"x">), $y)]
//                   >;
// 

// Lv 2

// y = alpha Ax + beta y
// currently assumes for vector dimensions that transa = 'N' and gets dimensions wrong otherwise
def gemv : CallBlasPattern<(Op $layout, $transa, $m, $n, $alpha, $A, $lda, $x, $incx, $beta, $y, $incy),
                  ["y"], [cblas_layout, trans,  len, len,  fp,   mld<["m", "n"]>,      vinc<["transa", "n", "m"]>,      fp,    vinc<["transa", "m", "n"]>],
                  [
                  /* alpha  */  (Seq<["Ax", "is_normal", "transa", "m", "n"]>
                      (BlasCall<"gemv"> $layout, $transa, $m, $n, Constant<"1.0">, $A, (ld $A, Char<"N">, $lda, $m, $m), $x, Constant<"0.0">, use<"Ax">, ConstantInt<1>),
                      (BlasCall<"dot"> (Rows $transa, $m, $n), (Shadow $y), use<"Ax">, ConstantInt<1>)),

                      //if (is_normal $transa) {
                      //  call sger(m, n, alpha, ya, incy, x, incx, Aa, lda)
                      //} else {
                      //  call sger(m, n, alpha, x, incx, ya, incy, Aa, lda)
                      //}
                  /* A      */  (BlasCall<"ger">  $layout, $m, $n, $alpha, (Rows $transa, (Concat (Shadow $y), $x), (Concat $x, (Shadow $y))), (Shadow $A)),
                  /* x      */  (BlasCall<"gemv"> $layout, transpose<"transa">, $m, $n, $alpha, $A, (ld $A, Char<"N">, $lda, $m, $m), (Shadow $y), Constant<"1.0">, (Shadow $x)),
                  /* beta   */  (BlasCall<"dot"> (Rows $transa, $m, $n), (Shadow $y), input<"y">),
                  /* y      */  (BlasCall<"scal"> (Rows $transa, $m, $n), $beta, (Shadow $y))
                  ]
                  >;

// x = Ax
// currently assumes for vector dimensions that transa = 'N' and gets dimensions wrong otherwise
def trmv : CallBlasPattern<(Op $layout, $transa, $diag, $n, $A, $lda, $x, $incx),
                  ["x"], [cblas_layout, trans,  diag, len,  mld<["diag", "n", "n"]>, vinc<["n"]>],
                  [
                  /* A      */  (AssertingInactiveArg), //(BlasCall<"ger">  $layout, $m, $n, $alpha, (Rows $transa, (Concat (Shadow $y), $x), (Concat $x, (Shadow $y))), (Shadow $A)),
                  /* x      */  (BlasCall<"trmv"> $layout, transpose<"transa">, $diag, $n, $A, (ld $A, Char<"N">, $lda, $n, $n), (Shadow $x))
                  ]
                  >;
// 
def ger : CallBlasPattern<(Op $layout, $m, $n, $alpha, $x, $incx, $y, $incy, $A, $lda),
                  ["A"],[cblas_layout, len, len, fp, vinc<["m"]>, vinc<["n"]>, mld<["m", "n"]>],
                  [
                  (AssertingInactiveArg),
                  (AssertingInactiveArg),
                  (AssertingInactiveArg),
                  (AssertingInactiveArg)
                  ]
                  >;
//(ld $A, $transa, $lda, $m, $k)
// if (cache_A) {
//   ld_A = (arg_transa == 'N') ? arg_k : arg_m;
// } else {
//   ld_A = arg_lda;
// }

def gemm : CallBlasPattern<(Op $layout, $transa, $transb, $m, $n, $k, $alpha, $A, $lda, $B, $ldb, $beta, $C, $ldc),
                  ["C"],
                  [cblas_layout, trans, trans, len, len, len, fp, mld<["transa", "m", "k"]>, mld<["transb", "k", "n"]>, fp, mld<["m", "n"]>],
                  [
                  
                  /* alpha */ (Seq<["AB", "product", "m", "n"]> 
                      (BlasCall<"gemm"> $layout, $transa, $transb, $m, $n, $k, Constant<"1.0">, $A, (ld $A, $transa, $lda, $k, $m), $B, (ld $B, $transb, $ldb, $k, $n), Constant<"0.0">, use<"AB">, $m),// TODO: check if last arg should be $m or $n
                      (FrobInnerProd<""> $m, $n, (Shadow $C), use<"AB">)),
                  /* A     */ (BlasCall<"gemm"> $layout, (Rows $transa, 
                                                    (Concat $transa, transpose<"transb">, $m, $k), 
                                                    (Concat $transb, $transa, $k, $m)),
                                                  $n, $alpha, 
                                                  (Rows $transa, 
                                                    (Concat (Shadow $C), $B, (ld $B, $transb, $ldb, $n, $k)), 
                                                    (Concat $B, (ld $B, $transb, $ldb, $n, $k), (Shadow $C))),
                                                  Constant<"1.0">, (Shadow $A)),

                  /* B     */ (BlasCall<"gemm"> $layout, (Rows $transb, 
                                                    (Concat transpose<"transa">, $transb, $k, $n),
                                                    (Concat $transb, $transa, $n, $k)),
                                                  $m, $alpha, 
                                                  (Rows $transb, 
                                                    (Concat $A, (ld $A, $transa, $lda, $k, $m), (Shadow $C)),
                                                    (Concat (Shadow $C), $A, (ld $A, $transa, $lda, $k, $m))),
                                                  Constant<"1.0">, (Shadow $B)),
                  /* beta  */ (FrobInnerProd<""> $m, $n, (Shadow $C), input<"C">),
                  /* C     */ (BlasCall<"lascl"> $layout, Char<"G">, ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, $beta, $m, $n, (Shadow $C), Alloca<1>)
                  ]
                  >;

def syrk : CallBlasPattern<(Op $layout, $uplo, $trans, $n, $k, $alpha, $A, $lda, $beta, $C, $ldc),
                  ["C"],
                  [cblas_layout, uplo, trans, len, len, fp, mld<["trans", "n", "k"]>, fp, mld<["n", "n"]>],
                  [
                  
                  /* alpha */ (AssertingInactiveArg), /*(Seq<["AB", "product", "m", "n"]> 
                      (BlasCall<"gemm"> $layout, $transa, $transb, $m, $n, $k, Constant<"1.0">, $A, (ld $A, $transa, $lda, $k, $m), $B, (ld $B, $transb, $ldb, $k, $n), Constant<"0.0">, use<"AB">, $m),// TODO: check if last arg should be $m or $n
                      (FrobInnerProd<""> $m, $n, (Shadow $C), use<"AB">)),*/
                  /* A     */ (AssertingInactiveArg), /*(BlasCall<"gemm"> $layout, (Rows $transa, 
                                                    (Concat $transa, transpose<"transb">, $m, $k), 
                                                    (Concat $transb, $transa, $k, $m)),
                                                  $n, $alpha, 
                                                  (Rows $transa, 
                                                    (Concat (Shadow $C), $B, (ld $B, $transb, $ldb, $n, $k)), 
                                                    (Concat $B, (ld $B, $transb, $ldb, $n, $k), (Shadow $C))),
                                                  Constant<"1.0">, (Shadow $A)),*/

                  /* beta  */ (AssertingInactiveArg), //(FrobInnerProd<""> $m, $n, (Shadow $C), input<"C">),
                  /* C     */ (AssertingInactiveArg), //(BlasCall<"lascl"> $layout, Char<"G">, ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, $beta, $m, $n, (Shadow $C), Alloca<1>)
                  ]
                  >;

def spmv : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $ap, $x, $incx, $beta, $y, $incy),
                  ["y"],
                  [cblas_layout, uplo, len, fp, ap<["n"]>, vinc<["n"]>, fp, vinc<["n"]>],
                  [
                  /* alpha */ (Seq<["y0", "triangular", "n"]>
                    (BlasCall<"spmv"> $layout, $uplo, $n, Constant<"1.0">, $ap, $x, Constant<"0.0">, use<"y0">, ConstantInt<1>),
                    (BlasCall<"dot"> $n, (Shadow $y), use<"y0">, ConstantInt<1>)),
                  /* ap    */ (Seq<[]>
                    (BlasCall<"spr2"> $layout, $uplo, $n, $alpha, $x, (Shadow $y), (Shadow $ap)),
                    (DiagUpdateSPMV<""> $uplo, $n, $alpha, $x, (Shadow $y), (Shadow $ap))),
                  /* x     */ (BlasCall<"spmv"> $layout, $uplo, $n, $alpha, $ap, (Shadow $y), Constant<"1.0">, (Shadow $x)),
                  /* beta  */ (BlasCall<"dot"> $n, (Shadow $y), input<"y">),
                  /* y     */ (BlasCall<"scal"> $n, $beta, (Shadow $y))
                  ]
                  >;

// B2 = inv(A^T) B
// dB = inv(A^T) dB2
// d(A^T) −= dB B2^T
def trtrs : CallBlasPattern<(Op $layout, $uplo, $trans, $diag, $n, $nrhs, $a, $lda, $b, $ldb, $info),
                  ["b"],
                  [cblas_layout, uplo, trans, diag, len, len, mld<["n", "n"]>, vinc<["n"]>, len],
                  [
                  /* a     */ (AssertingInactiveArg),
                  /* b     */ (AssertingInactiveArg),
                  ]
                  >;

def spr2 : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $x, $incx, $y, $incy, $ap),
                  ["ap"],
                  [cblas_layout, uplo, len, fp, vinc<["n"]>, vinc<["n"]>, ap<["n"]>],
                  [
                  /* alpha */ (AssertingInactiveArg),
                  /* x     */ (AssertingInactiveArg),
                  /* y     */ (AssertingInactiveArg),
                  /* ap    */ (AssertingInactiveArg),
                  ]
                  >;

// def spr : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $x, $incx, $v, $ld),
//                   [cblas_layout, uplo, len, fp, vinc, vld],
//                   []
//                   >;
// 
// def gbmv : CallBlasPattern<(Op $layout, $transa, $m, $n, $ml, $mu, $alpha, $v, $ld, $x, $incx, $beta, $y, $incy),
//                   [cblas_layout, trans, len, len, bw, bw, fp, vld, vinc, fp, vinc],
//                   []
//                   >;
// 
// def sbmv : CallBlasPattern<(Op $layout, $uplo, $n, $k, $alpha, $v, $ld, $x, $incx, $beta, $y, $incy),
//                   [cblas_layout, uplo, len, hbw, fp, vld, vinc, fp, vinc],
//                   []
//                   >;
// 
// def trmv : CallBlasPattern<(Op $layout, $uplo, $transa, $diag, $n, $v, $ld, $x, $incx),
//                   [cblas_layout, uplo, trans, diag, len, vld, vinc],
//                   []
//                   >;
// 
// def trsv : CallBlasPattern<(Op $layout, $uplo, $transa, $diag, $n, $v, $ld, $x, $incx),
//                   [cblas_layout, uplo, trans, diag, len, vld, vinc],
//                   []
//                   >;
// 
// def tbmv : CallBlasPattern<(Op $layout, $uplo, $transa, $diag, $n, $k, $v, $ld, $x, $incx),
//                   [cblas_layout, uplo, trans, diag, len, bw, vld, vinc],
//                   []
//                   >;
// 
// def tbsv : CallBlasPattern<(Op $layout, $uplo, $transa, $diag, $n, $k, $v, $ld, $x, $incx),
//                   [cblas_layout, uplo, trans, diag, len, bw, vld, vinc],
//                   []
//                   >;
// 
// // Lv 3
// def : CallBlasPattern<(Op $layout, $transa, $transb, $l, $n, $m, $alpha, $a, $lda, $b, $ldb, $beta, $c, $ldc),
//                   ["gemm"],
//                   [cblas_layout, trans, trans, len, len, len, fp, vld, vld, fp, vld],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $side, $uplo, $m, $n, $alpha, $a, $lda, $b, $ldb, $beta, $c, $ldc),
//                   ["symm"],
//                   [cblas_layout, side, uplo, len, len, fp, vld, vld, fp, vld],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $side, $uplo, $transa, $diag, $m, $n, $alpha, $a, $lda, $b, $ldb),
//                   ["trmm"],
//                   [cblas_layout, side, uplo, trans, diag, len, len, fp, vld, vld],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $side, $uplo, $transa, $diag, $m, $n, $alpha, $a, $lda, $b, $ldb),
//                   ["trsm"],
//                   [cblas_layout, side, uplo, trans, diag, len, len, fp, vld, vld],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $trans, $n, $k, $alpha, $a, $lda, $beta, $c, $ldc),
//                   ["syrk"],
//                   [cblas_layout, uplo, trans, len, len, fp, vld, fp, vld],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $trans, $n, $k, $alpha, $a, $lda, $b, $ldb, $beta, $c, $ldc),
//                   ["syr2k"],
//                   [cblas_layout, uplo, trans, len, len, fp, vld, vld, fp, vld],
//                   []
//                   >;
// 
