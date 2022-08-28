/*
* Design:
*/

class BLASType<int _n, bit _active> { 
  int nelem = _n; 
  bit active = _active;
}

def len   : BLASType<1, false>; // num of elements
def hbw   : BLASType<1, false>; // half matrix-bandwith}
def bw    : BLASType<1, false>; // matrix-bandwith}
def fp    : BLASType<1, true>;  // floating point
def ap    : BLASType<1, true>;  // packed array, alternative to vld
def vinc  : BLASType<2, true>;  // {vector, increment}
def vld   : BLASType<2, true>;  // {vector, leading dim}
def mld   : BLASType<2, true>;  // {matrix, leading dim}

class blas_modes<list<string> _modes> : BLASType<1, false> {
  list<string> modes = _modes;
}

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

// Lv 1 \wo rotations
def scal : CallBlasPattern<(Op $n, $alpha, $x, $incx),
                  [],[len, fp, vinc],
                  [
                  (b<"dot"> $n, $x, $incx, DiffeRet<"">, $incx), 
                  (b<"axpy"> $n, $alpha, DiffeRet<"">, $incx, $x, $incx)
                  ]
                  >;

def axpy : CallBlasPattern<(Op $n, $alpha, $x, $incx, $y, $incy),
                  ["y"],[len, fp, vinc, vinc],
                  [
                  (b<"dot"> $n, $x, $incx, DiffeRet<"">, $incy), 
                  (b<"scal"> $n, $alpha, DiffeRet<"">, $incy),
                  (b<"scal"> $n, $alpha, DiffeRet<"">, $incy) // TODO: fix
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
//                   [len, vinc],
//                   [(FDiv (b<"scal"> $n, DiffeRet<"">, $x, $incx), Ret<"">)]
//                   >;


// def copy : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
//                   [len, vinc, vinc],
//                   [
//                   (noop),
//                   (todo)
//                   ]
//                   >;
// 
// def swap : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
//                   [len, vinc, vinc],
//                   []
//                   >;

// def asum : CallBlasPattern<(Op $n, $x, $incx),
//                   [len, vinc],
//                   [(b<"scal"> $n, DiffeRet<"">, (sgn $x), $incx)]
//                   >;


// Lv 2

// y = alpha Ax + beta y
// def gemv : CallBlasPattern<(Op $layout, $transa, $m, $n, $alpha, $A, $lda, $x, $incx, $beta, $y, $incy),
//                   [cblas_layout, trans, len, len, fp, mld, vinc, fp, vinc],
//                   [
//                   /* alpha  */  (b<"dot"> DiffeRet<"">, (b<"gemv"> $A, $x)),
//                   /* A      */  //(),
//                   /* x      */  //(),
//                   /* beta   */  (b<"dot"> $y, DiffeRet<"">),
//                   /* y      */  (b<"scal"> $beta, DiffeRet<"">),
//                   ]
//                   >;
// 
// def ger : CallBlasPattern<(Op $layout, $m, $n, $alpha, $x, $incx, $y, $incy, $v, $ld),
//                   [cblas_layout, len, len, fp, vinc, vinc, vld],
//                   []
//                   >;

// def : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $v, $ld, $x, $incx, $beta, $y, $incy),
//                   ["spmv"],
//                   [cblas_layout, uplo, len, fp, vld, vinc, fp, vinc],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $x, $incx, $v, $ld),
//                   ["spr"],
//                   [cblas_layout, uplo, len, fp, vinc, vld],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $x, $incx, $y, $incy, $v, $ld),
//                   ["spr2"],
//                   [cblas_layout, uplo, len, fp, vinc, vinc, vld],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $transa, $m, $n, $ml, $mu, $alpha, $v, $ld, $x, $incx, $beta, $y, $incy),
//                   ["gbmv"],
//                   [cblas_layout, trans, len, len, bw, bw, fp, vld, vinc, fp, vinc],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $n, $k, $alpha, $v, $ld, $x, $incx, $beta, $y, $incy),
//                   ["sbmv"],
//                   [cblas_layout, uplo, len, hbw, fp, vld, vinc, fp, vinc],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $transa, $diag, $n, $v, $ld, $x, $incx),
//                   ["trmv"],
//                   [cblas_layout, uplo, trans, diag, len, vld, vinc],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $transa, $diag, $n, $v, $ld, $x, $incx),
//                   ["trsv"],
//                   [cblas_layout, uplo, trans, diag, len, vld, vinc],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $transa, $diag, $n, $k, $v, $ld, $x, $incx),
//                   ["tbmv"],
//                   [cblas_layout, uplo, trans, diag, len, bw, vld, vinc],
//                   []
//                   >;
// 
// def : CallBlasPattern<(Op $layout, $uplo, $transa, $diag, $n, $k, $v, $ld, $x, $incx),
//                   ["tbsv"],
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
