/*
* Design:
*/

class BLASType<int _n, bit _active> { 
  int nelem = _n; 
  bit active = _active;
}

def len   : BLASType<1, 0>; // num of elements
def hbw   : BLASType<1, 0>; // half matrix-bandwith}
def bw    : BLASType<1, 0>; // matrix-bandwith}
def fp    : BLASType<1, 1>;  // floating point
def ap    : BLASType<1, 1>;  // packed array, alternative to vld
def vinc  : BLASType<2, 1>;  // {vector, increment}
def vld   : BLASType<2, 1>;  // {vector, leading dim}
def mld   : BLASType<2, 1>;  // {matrix, leading dim}, assumes m, n as existing args

class blas_modes<list<string> _modes> : BLASType<1, 0> {
  list<string> modes = _modes;
}

// cblas_layout must(!) allways be called layout, or the world will burn
def cblas_layout  : blas_modes<["CblasRowMajor","CblasColMajor"]>;
def uplo          : blas_modes<["U","L"]> ;
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

class transpose<string _name> {
  string name = _name;
}


// General note: If return is scalar, return it. If return is vec, update it.

// Lv 1 \wo rotations
def scal : CallBlasPattern<(Op $n, $alpha, $x, $incx),
                  ["x"],[len, fp, vinc],
                  [
                  // dot must proceed scal, because scal modifies adj<"x">
                  (b<"dot"> $n, $x, $incx, adj<"x">, $incx),
                  (b<"scal"> $n, $alpha, adj<"x">, $incx)
                  ]
                  >;

def axpy : CallBlasPattern<(Op $n, $alpha, $x, $incx, $y, $incy),
                  ["y"],[len, fp, vinc, vinc],
                  [
                  // dot must proceed scal, because scal modifies adj<"x">
                  (b<"axpy"> $n, $alpha, adj<"x">, $incx, adj<"y">, $incy),
                  (b<"scal"> $n, $alpha, adj<"x">, $incy),
                  (noop) // y = (Ax) + y, so nothing to do here
                  ]
                  >;

def dot : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
                  [],[len, vinc, vinc],
                  [
                  (b<"axpy"> $n, DiffeRet<"">, $y, $incy, $x, $incx), 
                  (b<"axpy"> $n, DiffeRet<"">, $x, $incx, $y, $incy)
                  ]
                  >;

// def nrm2 : CallBlasPattern<(Op $n, $x, $incx),
//                   [],[len, vinc],
//                   [(FDiv (b<"scal"> $n, DiffeRet<"">, $x, $incx), Ret<"">)]
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
//                   [(b<"axpy"> DiffeRet<"">, sign(<input<"x">), $y)]
//                   >;
// 

// Lv 2

// y = alpha Ax + beta y
def gemv : CallBlasPattern<(Op $layout, $transa, $m, $n, $alpha, $A, $lda, $x, $incx, $beta, $y, $incy),
                  ["y"], [cblas_layout, trans,  len, len,  fp,   mld,      vinc,      fp,    vinc],
                  [
                  /* alpha  */  (noop), //(b<"dot"> $y, (b<"gemv"> $A, $x)),
                  /* A      */  (b<"ger">  $layout, $m, $n, $alpha, adj<"y">, $incy, $x, $incx, $A, $lda),
                  /* x      */  (noop), //(b<"gemv"> $layout, $m, $n, $alpha, $A, $lda, $y, $incy, $x),
                  /* beta   */  (noop),//(b<"dot"> $y, DiffeRet<"">),
                  /* y      */  (noop)//(b<"scal"> $beta, DiffeRet<"">),
                  ]
                  >;
// 
def ger : CallBlasPattern<(Op $layout, $m, $n, $alpha, $x, $incx, $y, $incy, $A, $lda),
                  ["A"],[cblas_layout, len, len, fp, vinc, vinc, mld],
                  [
                  (noop),
                  (noop),
                  (noop),
                  (noop)
                  ]
                  >;

def gemm : CallBlasPattern<(Op $layout, $transa, $transb, $m, $n, $k, $alpha, $A, $lda, $B, $ldb, $beta, $C, $ldc),
                  ["C"],
                  [cblas_layout, trans, trans, len, len, len, fp, mld, mld, fp, mld],
                  [
                  /* alpha */ (noop),
                  /* A     */ (b<"gemm"> $layout, $transa, transpose<"transb">, $m, $k, $n, $alpha, adj<"C">, $ldc, $B, $ldb, $beta, adj<"A">, $ldc),
                  /* B     */ (b<"gemm"> $layout, transpose<"transa">, $transb, $m, $n, $k, $alpha, $A, $lda, adj<"C">, $ldc, $beta, adj<"B">, $ldb),
                  /* beta  */ (noop),
                  /* C     */ (noop) // check that beta always zero! Then set adj<"C"> = 0 (if beta = 1, do nothing). If beta anything else, use scal on dc
                  ]
                  >;

// def spmv : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $v, $ld, $x, $incx, $beta, $y, $incy),
//                   ["spmv"],
//                   [cblas_layout, uplo, len, fp, vld, vinc, fp, vinc],
//                   []
//                   >;
// 
// def spr : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $x, $incx, $v, $ld),
//                   [cblas_layout, uplo, len, fp, vinc, vld],
//                   []
//                   >;
// 
// def spr2 : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $x, $incx, $y, $incy, $v, $ld),
//                   [cblas_layout, uplo, len, fp, vinc, vinc, vld],
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
