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
def sgn  : MagicInst; // only the sign
def zero : MagicInst; // zeros of given length
def todo : MagicInst; // TODO
def tp   : MagicInst; // transpose the trans param.
def noop : MagicInst; // gradient is zero 
def inactive : MagicInst; // like noop, but assert it's inactive
def Rows : MagicInst; // given a transpose, normal rows, normal cols get the true rows, aka normal rows if N else normal cols

// if !cache_A, then just use $lda.
// if cache_A, then check $transa.
// if $transa = 'N', then use $m.
// else use $n
def ld : MagicInst; // (ld $A, $transa $lda, $m, $n): 

// placeholder name
class b<string _s> {
  string s = _s;
}

// only applicable to arguments being modified
// will cache the input values to be used later
class input<string _name> {
  string name = _name;
}

class adj<string _name> {
  string name = _name;
}

class Constant<string _value> {
  string value = _value;
}
class Char<string _value> {
  string value = _value;
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

class FrobInnerProd<string _tmp>;

class DiagUpdateSPMV<string _tmp>;

// General note: If return is scalar, return it. If return is vec, update it.

// Lv 1 \wo rotations
def scal : CallBlasPattern<(Op $n, $alpha, $x, $incx),
                  ["x"],[len, fp, vinc<["n"]>],
                  [
                  // dot must proceed scal, because scal modifies adj<"x">
                  (b<"dot"> $n, $x, $incx, adj<"x">, $incx),
                  (b<"scal"> $n, $alpha, adj<"x">, $incx)
                  ]
                  >;

// def lacpy : CallBlasPattern<(Op $layout, $m, $n, $A, $lda, $B, $ldb),
//                   ["B"],[cblas_layout, len, len, mld<["m", "n"]>, mld<["m", "n"]>],
//                   [
//                   (inactive), // from
//                   (inactive), // to
//                   ]
//                   >;

def lascl : CallBlasPattern<(Op $layout, $type, $kl, $ku, $cfrom, $cto, $m, $n, $A, $lda, $info),
                  ["A"],[cblas_layout, uplo, len, len, fp, fp, len, len, mld<["m", "n"]>, len],
                  [
                  (inactive), // cfrom
                  (inactive), // cto
                  (inactive)  // A
                  ]
                  >;

def axpy : CallBlasPattern<(Op $n, $alpha, $x, $incx, $y, $incy),
                  ["y"],[len, fp, vinc<["n"]>, vinc<["n"]>],
                  [
                  // dot must proceed scal, because scal modifies adj<"x">
                  (inactive),
                  (inactive),//(b<"scal"> $n, $alpha, adj<"x">, $incy),
                  (inactive) // y = (Ax) + y, so nothing to do here
                  ]
                  >;

def dot : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
                  [],[len, vinc<["n"]>, vinc<["n"]>],
                  [
                  (b<"axpy"> $n, DiffeRet, $y, $incy, adj<"x">, $incx), 
                  (b<"axpy"> $n, DiffeRet, $x, $incx, adj<"y">, $incy)
                  ]
                  >;

// def nrm2 : CallBlasPattern<(Op $n, $x, $incx),
//                   [],[len, vinc],
//                   [(FDiv (b<"scal"> $n, DiffeRet, $x, $incx), Ret<"">)]
//                   >;


// def copy : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
//                   ["y"],[len, vinc, vinc],
//                   [
//                   (noop),// copy moves x into y, so x is never modified.
//                   (b<"axpy"> $n, Constant<"1.0">, adj<"y">, $incy, adj<"x">, $incx)
//                   ]
//                   >;

// def swap : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
//                   ["x","y"],[len, vinc, vinc],
//                   [
//                   (b<"swap"> $n, adj<"x">, $incx, adj<"y">, $incy),
//                   (noop),// one swap is enough, will only be used iff x and y active anyway
//                   ]
//                   >;

// def asum : CallBlasPattern<(Op $n, $x, $incx),
//                   [],[len, vinc],
//                   [(b<"axpy"> DiffeRet, sign(<input<"x">), $y)]
//                   >;
// 

// Lv 2

// y = alpha Ax + beta y
// currently assumes for vector dimensions that transa = 'N' and gets dimensions wrong otherwise
def gemv : CallBlasPattern<(Op $layout, $transa, $m, $n, $alpha, $A, $lda, $x, $incx, $beta, $y, $incy),
                  ["y"], [cblas_layout, trans,  len, len,  fp,   mld<["m", "n"]>,      vinc<["transa", "n", "m"]>,      fp,    vinc<["transa", "m", "n"]>],
                  [
                  /* alpha  */  (Seq<["Ax", "is_normal", "transa", "m", "n"]>
                      (b<"gemv"> $layout, $transa, $m, $n, Constant<"1.0">, $A, (ld $A, $transa, $lda, $m, $n), $x, $incx, Constant<"0.0">, use<"Ax">, ConstantInt<1>),
                      (b<"dot"> (Rows $transa, $m, $n), adj<"y">, $incy, use<"Ax">, ConstantInt<1>)),
                  /* A      */  (b<"ger">  $layout, $m, $n, $alpha, adj<"y">, $incy, $x, $incx, adj<"A">, $lda),
                  /* x      */  (b<"gemv"> $layout, transpose<"transa">, $m, $n, $alpha, $A, (ld $A, $transa, $lda, $m, $n), adj<"y">, $incy, Constant<"1.0">, adj<"x">, $incx),
                  /* beta   */  (b<"dot"> (Rows $transa, $m, $n), adj<"y">, $incy, input<"y">, $incy),
                  /* y      */  (b<"scal"> (Rows $transa, $m, $n), $beta, adj<"y">, $incy)
                  ]
                  >;
// 
def ger : CallBlasPattern<(Op $layout, $m, $n, $alpha, $x, $incx, $y, $incy, $A, $lda),
                  ["A"],[cblas_layout, len, len, fp, vinc<["m"]>, vinc<["n"]>, mld<["m", "n"]>],
                  [
                  (inactive),
                  (inactive),
                  (inactive),
                  (inactive)
                  ]
                  >;
//(ld $A, $transa, $lda, $m, $k)
// if (cache_A) {
//   ld_A = (arg_transa == 'N') ? arg_m : arg_k;
// } else {
//   ld_A = arg_lda;
// }

def gemm : CallBlasPattern<(Op $layout, $transa, $transb, $m, $n, $k, $alpha, $A, $lda, $B, $ldb, $beta, $C, $ldc),
                  ["C"],
                  [cblas_layout, trans, trans, len, len, len, fp, mld<["transa", "m", "k"]>, mld<["transb", "k", "n"]>, fp, mld<["m", "n"]>],
                  [
                  
                  /* alpha */ (Seq<["AB", "product", "m", "n"]> 
                      (b<"gemm"> $layout, $transa, $transb, $m, $n, $k, Constant<"1.0">, $A, (ld $A, $transa, $lda, $m, $k), $B, (ld $B, $transb, $ldb, $k, $n), Constant<"0.0">, use<"AB">, $m),// TODO: check if last arg should be $m or $n
                      (FrobInnerProd<""> $m, $n, adj<"C">, $ldc, use<"AB">)),
                  /* A     */ (b<"gemm"> $layout, $transa, transpose<"transb">, $m, $k, $n, $alpha, adj<"C">, $ldc, $B, (ld $B, $transb, $ldb, $k, $n), $beta, adj<"A">, $lda),
                  /* B     */ (b<"gemm"> $layout, transpose<"transa">, $transb, $k, $n, $m, $alpha, $A, (ld $A, $transa, $lda, $m, $k), adj<"C">, $ldc, $beta, adj<"B">, $ldb),
                  /* beta  */ (FrobInnerProd<""> $m, $n, adj<"C">, $ldc, input<"C">),
                  /* C     */ (b<"lascl"> $layout, Char<"G">, ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, $beta, $m, $n, adj<"C">, $ldc, ConstantInt<0>)
                  ]
                  >;

def spmv : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $ap, $x, $incx, $beta, $y, $incy),
                  ["y"],
                  [cblas_layout, uplo, len, fp, ap<["n"]>, vinc<["n"]>, fp, vinc<["n"]>],
                  [
                  /* alpha */ (Seq<["y0", "triangular", "n"]>
                    (b<"spmv"> $layout, $uplo, $n, Constant<"1.0">, $ap, $x, $incx, Constant<"0.0">, use<"y0">, ConstantInt<1>),
                    (b<"dot"> $n, adj<"y">, $incy, use<"y0">, ConstantInt<1>)),
                  /* ap    */ (Seq<[]>
                    (b<"spr2"> $layout, $uplo, $n, $alpha, $x, $incx, adj<"y">, $incy, adj<"ap">),
                    (DiagUpdateSPMV<""> $uplo, $n, $alpha, $x, $incx, adj<"y">, $incy, adj<"ap">)),
                  /* x     */ (b<"spmv"> $layout, $uplo, $n, $alpha, $ap, adj<"y">, $incy, Constant<"1.0">, adj<"x">, $incx),
                  /* beta  */ (b<"dot"> $n, adj<"y">, $incy, input<"y">, $incy),
                  /* y     */ (b<"scal"> $n, $beta, adj<"y">, $incy)
                  ]
                  >;

def spr2 : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $x, $incx, $y, $incy, $ap),
                  ["ap"],
                  [cblas_layout, uplo, len, fp, vinc<["n"]>, vinc<["n"]>, ap<["n"]>],
                  [
                  /* alpha */ (inactive),
                  /* x     */ (inactive),
                  /* y     */ (inactive),
                  /* ap    */ (inactive),
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
