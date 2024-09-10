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
def info          : blas_modes<[]>;


class CallBlasPattern<dag patternToMatch, list<string> mutables, list<BLASType> paramTypes, 
                      list<dag> resultOps, dag forwardOps = (AssertingInactiveArg)> {
  dag PatternToMatch = patternToMatch;
  list<BLASType> inputTypes = paramTypes;
  list<dag> ArgDerivatives = resultOps;
  list<string> mutable = mutables;
  dag ArgDuals = forwardOps;
}


class MagicInst : Inst<"blas">;
def Rows : MagicInst; // given a transpose, normal rows, normal cols get the true rows, aka normal rows if N else normal cols
def Concat : MagicInst;

def ShadowNoInc : MagicInst;

class Binop<string _s, list<string> _tys> {
  string s = _s;
  list<string> tys = _tys;
}
def BXor : Binop<"Xor", ["intType", "intType"]>;
def BAnd : Binop<"And", ["intType", "intType"]>;
def Not : Binop<"Not", ["intType", "intType"]>;
def Sub : Binop<"Sub", ["intType", "intType"]>;
def Add : Binop<"Add", ["intType", "intType"]>;
def Mul : Binop<"Mul", ["intType", "intType"]>;
def BFMul : Binop<"FMul", ["fpType", "fpType"]>;
def BFDiv : Binop<"FDiv", ["fpType", "fpType"]>;
def FSelect : Binop<"Select", ["Builder2.getInt1Ty()", "fpType", "fpType"]>;
def ISelect : Binop<"Select", ["Builder2.getInt1Ty()", "intType", "intType"]>;

class BIntrinsic<string name_, list<string> _tys, list<int> types_ = [0]>  {
  string s = name_;
  list<string> tys = _tys;
  list<int> types = types_;
}

def BSqrt : BIntrinsic<"sqrt", ["fpType"]>;

class IntMatchers<string _inty, string _outty, list<string> _before, list<string> _after> {
  string inty = _inty;
  string outty = _outty;
  list<string> before = _before;
  list<string> after = _after;
}

def side_to_trans : IntMatchers<
    "charType", "charType",
    ["'l'", "'L'", "'R'", "'r'"],
    ["'n'", "'N'", "'T'", "'t'"]
    >;

def is_diag_int : IntMatchers<
    "charType", "intType",
    ["'u'", "'U'", "'N'", "'n'"],
    ["1", "1", "0", "0"]
    >;

def is_left : MagicInst;
def is_lower : MagicInst;
def is_nonunit : MagicInst;

def First : MagicInst;
def Lookup : MagicInst;
def LoadLookup : MagicInst;

class FAdd<string _tmp=""> {
  string unused = _tmp;
}

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

// only applicable to triangular matricies, like a regular use of $A
// except that the non-set terms are assured to be set to 0
class zero_cached<string _name> {
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

class Seq<list<string> _args = [], list<string> _vars = [], bit _start = 1> {
  list<string> args = _args;
  list<string> vars = _vars;
  bit start = _start;
}

class CopyLowerToUpper<string _tmp> {
  string unused = _tmp;
}

class For<string idx_, bit offset_> {
  string idx = idx_;
  bit offset = offset_;
}

class FirstUse<string var_> {
  string var = var_;
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

// x *= alpha
// FWD: x2 = x * alpha, -> fwd is dx2 = dx * alpha + x * dalpha
// Lv 1 \wo rotations
def scal : CallBlasPattern<(Op $n, $alpha, $x, $incx),
                  ["x"],[len, fp, vinc<["n"]>],
                  [
                  // dot must preceed scal, because scal modifies (Shadow $x)
                  (BlasCall<"dot"> $n, $x, (Shadow $x)),
                  (BlasCall<"scal"> $n, $alpha, (Shadow $x))
                  ],
                  (Seq<[], [], 1> (BlasCall<"scal"> $n, $alpha, (Shadow $x)), (BlasCall<"axpy"> $n, (Shadow $alpha), $x, (Shadow $x)))
                  >;

def lacpy : CallBlasPattern<(Op $layout, $uplo, $m, $n, $A, $lda, $B, $ldb),
                  ["B"],[cblas_layout, uplo, len, len, mld<["m", "n"]>, mld<["m", "n"]>],
                  [
                  (AssertingInactiveArg), // from
                  (AssertingInactiveArg), // to
                  ]
                  >;

def lascl : CallBlasPattern<(Op $layout, $type, $kl, $ku, $cfrom, $cto, $m, $n, $A, $lda, $info),
                  ["A"],[cblas_layout, uplo, len, len, fp, fp, len, len, mld<["m", "n"]>, info],
                  [
                  (AssertingInactiveArg), // cfrom
                  (AssertingInactiveArg), // cto
                  (AssertingInactiveArg)  // A
                  ]
                  >;

// y = alpha * x + y
// FWD: dy = alpha * dx + dalpha * x + dy
def axpy : CallBlasPattern<(Op $n, $alpha, $x, $incx, $y, $incy),
                  ["y"],[len, fp, vinc<["n"]>, vinc<["n"]>],
                  [
                  (BlasCall<"dot"> $n, (Shadow $y), $x),
                  (BlasCall<"axpy"> $n, $alpha, (Shadow $y), (Shadow $x)),
                  (InactiveArg) // y = alpha*x + y, so nothing to do here
                  ],
                  (Seq<[], [], 1> (BlasCall<"axpy"> $n, $alpha, (Shadow $x), (Shadow $y)), (BlasCall<"axpy"> $n, (Shadow $alpha), $x, (Shadow $y)))
                  >;

// x * y
def dot : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
                  [],[len, vinc<["n"]>, vinc<["n"]>],
                  [
                  (BlasCall<"axpy"> $n, DiffeRet, $y, (Shadow $x)),
                  (BlasCall<"axpy"> $n, DiffeRet, $x, (Shadow $y)),
                  ],
                  (FAdd<""> (BlasCall<"dot"> $n, (Shadow $x), $y), (BlasCall<"dot"> $n, $x, (Shadow $y)))
                  >;

def nrm2 : CallBlasPattern<(Op $n, $x, $incx),
                  [],[len, vinc<["n"]>],
                  [
                  (BlasCall<"axpy"> $n, (BFDiv DiffeRet, (BlasCall<"nrm2"> $n, $x)), $x, (Shadow $x))
                  ],
		              (BFDiv (BSqrt (BlasCall<"dot"> $n, $x, (Shadow $x))), (BlasCall<"nrm2"> $n, $x))
                  >;


def copy : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
                  ["y"],[len, vinc<["n"]>, vinc<["n"]>],
                  [
                  (InactiveArg),// copy moves x into y, so x is never modified.
                  (BlasCall<"axpy"> $n, Constant<"1.0">, (Shadow $y), (Shadow $x))
                  ],
                  (Seq<[], ["beta1"], 1>
                    (BlasCall<"copy"> (FirstUse<"beta1"> $n, $n), (Shadow $x), (Shadow $y)),
                    (FirstUse<"beta1"> (BlasCall<"scal"> $n, Constant<"0">, (Shadow $y)))
                  )
                  >;

// def swap : CallBlasPattern<(Op $n, $x, $incx, $y, $incy),
//                   ["x","y"],[len, vinc, vinc],
//                   [
//                   (BlasCall<"swap"> $n, (Shadow $x), $incx, (Shadow $y), $incy),
//                   (InactiveArg),// one swap is enough, will only be used iff x and y active anyway
//                   ]
//                   >;

def asum : CallBlasPattern<(Op $n, $x, $incx),
                  [],[len, vinc<["n"]>],
                  [(AssertingInactiveArg)]
                  >;

// Lv 2

// y = alpha Ax + beta y
// FWD: dy = dalpha Ax + alpha dA x + alpha A dx + dbeta y + beta dy
// currently assumes for vector dimensions that transa = 'N' and gets dimensions wrong otherwise
def gemv : CallBlasPattern<(Op $layout, $transa, $m, $n, $alpha, $A, $lda, $x, $incx, $beta, $y, $incy),
                  ["y"], [cblas_layout, trans,  len, len,  fp,   mld<["m", "n"]>,      vinc<["transa", "n", "m"]>,      fp,    vinc<["transa", "m", "n"]>],
                  [
                  /* alpha  */  (Seq<["Ax", "is_normal", "transa", "m", "n"], [], 1>
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
                  ],
                  (Seq<[], ["beta1"], 1>
                    (BlasCall<"axpy"> (Rows $transa, $m, $n), (Shadow $beta), $y, (Shadow $y)),
                    (BlasCall<"gemv"> $layout, $transa, $m, $n, $alpha,                   $A, (ld $A, Char<"N">, $lda, $m, $m),  (Shadow $x), (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $y)),
                    (BlasCall<"gemv"> $layout, $transa, $m, $n, $alpha,          (Shadow $A),                                             $x, (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $y)),
                    (BlasCall<"gemv"> $layout, $transa, $m, $n, (Shadow $alpha),          $A, (ld $A, Char<"N">, $lda, $m, $m),           $x, (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $y)),
                    (FirstUse<"beta1"> (BlasCall<"scal"> (Rows $transa, $m, $n), $beta, (Shadow $y)))
                  )
                  >;

// x = Ax
// currently assumes for vector dimensions that transa = 'N' and gets dimensions wrong otherwise
def trmv : CallBlasPattern<(Op $layout, $uplo, $trans, $diag, $n, $A, $lda, $x, $incx),
                  ["x"], [cblas_layout, uplo, trans,  diag, len,  mld<["uplo", "n", "n"]>, vinc<["n"]>],
                  [
                  /* A      */  (For<"i", 1> (ISelect (is_lower $uplo), (Sub $n, (is_diag_int $diag)), $n),
                    (BlasCall<"axpy"> 
                    (ISelect (is_lower $uplo), 
                        (Add (Sub (Sub $n, (is_diag_int $diag)), $i), ConstantInt<1>),
                        (Sub $i, (is_diag_int $diag))
                    ),
                    (LoadLookup $layout, (Rows $trans, input<"x">, (Shadow $x)), (Sub $i, ConstantInt<1>)),
                    (Lookup $layout, (Rows $trans, (Shadow $x), input<"x">), (ISelect (is_lower $uplo), (Sub (Add $i, (is_diag_int $diag)), ConstantInt<1>), ConstantInt<0>)),
                    (First (Lookup $layout, (Shadow $A), (ISelect (is_lower $uplo), (Sub (Add $i, (is_diag_int $diag)), ConstantInt<1>), ConstantInt<0>), (Sub $i, ConstantInt<1>))),
                    ConstantInt<1>
                  )
                  ),
                  /* x      */  (BlasCall<"trmv"> $layout, $uplo, transpose<"trans">, $diag, $n, $A, (ld $A, Char<"N">, $lda, $n, $n), (Shadow $x))
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

// C := alpha*op( A )*op( B ) + beta*C
def gemm : CallBlasPattern<(Op $layout, $transa, $transb, $m, $n, $k, $alpha, $A, $lda, $B, $ldb, $beta, $C, $ldc),
                  ["C"],
                  [cblas_layout, trans, trans, len, len, len, fp, mld<["transa", "m", "k"]>, mld<["transb", "k", "n"]>, fp, mld<["m", "n"]>],
                  [
                  
                  /* alpha */ (Seq<["AB", "product", "m", "n"], [], 1> 
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
                  ],
                  // FWD: dC = dalpha A B + alpha dA B + alpha A dB + dbeta C + beta dC 
                  (Seq<[], ["beta1"], 1>
                    (BlasCall<"axpy"> (AssertingInactiveArg), (Shadow $beta), $C, (Shadow $C)),
                    (BlasCall<"gemm"> $layout, $transa, $transb, $m, $n, $k, $alpha, $A, (ld $A, $transa, $lda, $k, $m),          (Shadow $B),                        (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $C)),
                    (BlasCall<"gemm"> $layout, $transa, $transb, $m, $n, $k, $alpha, (Shadow $A),                                 $B, (ld $B, $transb, $ldb, $n, $k), (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $C)),
                    (BlasCall<"gemm"> $layout, $transa, $transb, $m, $n, $k, (Shadow $alpha), $A, (ld $A, $transa, $lda, $k, $m), $B, (ld $B, $transb, $ldb, $n, $k), (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $C)),
                    (FirstUse<"beta1"> (BlasCall<"lascl"> $layout, Char<"G">, ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, $beta, $m, $n, (Shadow $C), Alloca<1>))
                  )
                  >;

//   B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
def trmm : CallBlasPattern<(Op $layout, $side, $uplo, $transa, $diag, $m, $n, $alpha, $A, $lda, $B, $ldb),
                  ["B"],
                  [cblas_layout, side, uplo, trans, diag, len, len, fp, mld<["side", "m", "n"]>, mld<["m","n"]>],
                  [
                    /*alpha*/ (AssertingInactiveArg),
                    /* A      */  (For<"i", 1> (Sub (ISelect (is_left $side), $m, $n), (ISelect (is_lower $uplo), (is_diag_int $diag), ConstantInt<0>)),
                    (BlasCall<"gemv">
                    $layout,
                    (side_to_trans $side),
                    (ISelect (is_left $side),
                        (Concat 
                            (ISelect (is_lower $uplo),
                                     (Sub $m, (Sub (Add $i, (is_diag_int $diag)), ConstantInt<1>)),
                                     (Sub $i, (is_diag_int $diag))),
                          $n),
                        (Concat $m,
                            (ISelect (is_lower $uplo),
                                     (Sub $n, (Sub (Add $i, (is_diag_int $diag)), ConstantInt<1>)),
                                     (Sub $i, (is_diag_int $diag)))
                        )
                    ),
                    $alpha,
                    (Lookup $layout, (ISelect (BXor (is_left $side), (Not (Rows $transa))), (Shadow $B), (Concat input<"B">, $m)),
                        (ISelect (BAnd (is_left $side), (is_lower $uplo)), (Sub (Add $i, (is_diag_int $diag)), ConstantInt<1>), ConstantInt<0>),
                        (ISelect (BAnd (Not (is_left $side)), (is_lower $uplo)), (Sub (Add $i, (is_diag_int $diag)), ConstantInt<1>), ConstantInt<0>)),
                    (First (Lookup $layout, (ISelect (BXor (is_left $side), (Not (Rows $transa))), (Concat input<"B">, $m), (Shadow $B)),
                        (ISelect (is_left $side), (Sub $i, ConstantInt<1>), ConstantInt<0>),
                        (ISelect (is_left $side), ConstantInt<0>, (Sub $i, ConstantInt<1>)))),
                    (ISelect (is_left $side), (ISelect (BXor (is_left $side), (Not (Rows $transa))), $m, $ldb), ConstantInt<1>),
                    Constant<"1">,
                    (First (Lookup $layout, (Shadow $A),
                        (ISelect (is_lower $uplo), (Sub (Add $i, (is_diag_int $diag)), ConstantInt<1>), ConstantInt<0>),
                        (Sub $i, ConstantInt<1>))),
                    ConstantInt<1>
                    )
                    ),
                    /* B   */ (BlasCall<"trmm"> $layout, $side, $uplo, transpose<"transa">, $diag, $m, $n, $alpha, $A, (ld $A, (side_to_trans $side), $lda, $n, $m), (Shadow $B))
                  ]
                  >;

def symm: CallBlasPattern<(Op $layout, $side, $uplo, $m, $n, $alpha, $A, $lda, $B, $ldb, $beta, $C, $ldc),
                          ["C"],
                          [cblas_layout, side, uplo, len, len, fp, mld<["side", "m", "n"]>, mld<["m", "n"]>, fp, mld<["m", "n"]>],
                   [
                        /*alpha*/ (AssertingInactiveArg),
                        /*A*/ (Seq<["tmp", "is_left", "side", "m", "n"], [], 1>
                            (BlasCall<"copy">
                              (ISelect (is_left $side), $m, $n),
                              (First (Shadow $A)),
                              (Add $lda, ConstantInt<1>),
                              use<"tmp">,
                              ConstantInt<1>
                            ),
                            (BlasCall<"syr2k">
                            $layout,
                            $uplo,
                            (side_to_trans $side),
                            (ISelect (is_left $side), $m, $n),
                            (ISelect (is_left $side), $n, $m),
                            $alpha,
                            $B, (ld $B, Char<"N">, $ldb, $m, $m),
                            (Shadow $C),
                            Constant<"1.0">, 
                            (Shadow $A)
                            ),
                            (BlasCall<"axpy"> 
                              (ISelect (is_left $side), $m, $n), 
                              Constant<"-1">,
                              (First (Shadow $A)),
                              (Add $lda, ConstantInt<1>),
                              use<"tmp">,
                              ConstantInt<1>
                            ),
                            (BlasCall<"axpy"> 
                              (ISelect (is_left $side), $m, $n), 
                              Constant<"0.5">,
                              use<"tmp">,
                              ConstantInt<1>,
                              (First (Shadow $A)),
                              (Add $lda, ConstantInt<1>)
                            )
                        ),
                        /*B*/ (BlasCall<"symm"> $layout, $side, $uplo, $m, $n, $alpha, $A, (ld $A, (side_to_trans $side), $lda, $n, $m), (Shadow $C), Constant<"1">, (Shadow $B)),
                        /*beta*/ (AssertingInactiveArg),
                        /*C*/ (BlasCall<"lascl"> $layout, Char<"G">, ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, $beta, $m, $n, (Shadow $C), Alloca<1>)
                   ],
                  // FWD: dC = dalpha A B + alpha dA B + alpha A dB + dbeta C + beta dC 
                  (Seq<[], ["beta1"], 1>
                    (BlasCall<"axpy"> (AssertingInactiveArg), (Shadow $beta), $C, (Shadow $C)),
                    (BlasCall<"symm"> $layout, $side, $uplo, $m, $n, $alpha, $A, (ld $A, (side_to_trans $side), $lda, $n, $m),          (Shadow $B),                        (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $C)),
                    (BlasCall<"symm"> $layout, $side, $uplo, $m, $n, $alpha, (Shadow $A),                                 $B, (ld $B, Char<"N">, $ldb, $m, $m), (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $C)),
                    (BlasCall<"symm"> $layout, $side, $uplo, $m, $n, (Shadow $alpha), $A, (ld $A, (side_to_trans $side), $lda, $n, $m), $B, (ld $B, Char<"N">, $ldb, $m, $m), (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $C)),
                    (FirstUse<"beta1"> (BlasCall<"lascl"> $layout, Char<"G">, ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, $beta, $m, $n, (Shadow $C), Alloca<1>))
                  )
                  >;

def syr2k : CallBlasPattern<(Op $layout, $uplo, $trans, $n, $k, $alpha, $A, $lda, $B, $ldb, $beta, $C, $ldc),
                  ["C"],
                  [cblas_layout, uplo, trans, len, len, fp, mld<["trans", "n", "k"]>, mld<["trans", "n", "k"]>, fp, mld<["n", "n"]>],
                  [
                        /*alpha*/ (AssertingInactiveArg),
                        /*A*/ (AssertingInactiveArg),
                        /*B*/ (AssertingInactiveArg),
                        /*beta*/ (AssertingInactiveArg),
                        /*C*/ (AssertingInactiveArg),
                  ]
                  >;

def syrk : CallBlasPattern<(Op $layout, $uplo, $trans, $n, $k, $alpha, $A, $lda, $beta, $C, $ldc),
                  ["C"],
                  [cblas_layout, uplo, trans, len, len, fp, mld<["trans", "n", "k"]>, fp, mld<["n", "n"]>],
                  [
                  
                  /* alpha */ (AssertingInactiveArg), /*(Seq<["AB", "product", "m", "n"], [], true> 
                      (BlasCall<"gemm"> $layout, $transa, $transb, $m, $n, $k, Constant<"1.0">, $A, (ld $A, $transa, $lda, $n, $k), $B, (ld $B, $transb, $ldb, $k, $n), Constant<"0.0">, use<"AB">, $m),// TODO: check if last arg should be $m or $n
                      (FrobInnerProd<""> $m, $n, (Shadow $C), use<"AB">)),*/
                  /* A     */
                  (Seq<[], [], 1>
                    (BlasCall<"symm">
                        $layout,
                        (Rows $trans, Char<"l">, Char<"r">),
                        $uplo,
                        (Rows $trans,
                            (Concat $n, $k),
                            (Concat $k, $n)),
                        $alpha,
                        (Shadow $C),
                        $A, (ld $A, $trans, $lda, $n, $k),
                        Constant<"1">,
                        (Shadow $A)
                    ),
                    (For<"i", 0> $n,
                        (BlasCall<"axpy">
                            $k,
                            (BFMul $alpha, (LoadLookup $layout, (Shadow $C), $i, $i)),
                            (First
                                (Lookup $layout,
                                    (Concat $A, (ld $A, $trans, $lda, $n, $k)),
                                    (Rows $trans, $i, ConstantInt<0>),
                                    (Rows $trans, ConstantInt<0>, $i)
                                )
                            ),
                            (Rows $trans, (ld $A, $trans, $lda, $n, $k), ConstantInt<1>),
                            (First
                                (Lookup $layout,
                                    (Shadow $A),
                                    (Rows $trans, $i, ConstantInt<0>),
                                    (Rows $trans, ConstantInt<0>, $i)
                                )
                            ),
                            (Rows $trans, $lda, ConstantInt<1>)
                          )
                    )
                  ),
                  /* beta  */ (AssertingInactiveArg),
                  /* C     */ (BlasCall<"lascl"> $layout, $uplo, ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, $beta, $n, $n, (Shadow $C), Alloca<1>)
                  ],
                  (Seq<[], ["beta1"], 1>
                    (BlasCall<"axpy"> (AssertingInactiveArg), (Shadow $beta), $C, (Shadow $C)),
                    (BlasCall<"syr2k"> $layout, $uplo, $trans, $n, $k, $alpha, $A, $lda, (Shadow $A), (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $C)),
                    (BlasCall<"syrk"> $layout, $uplo, $trans, $n, $k, (Shadow $alpha), $A, $lda, (FirstUse<"beta1"> $beta, Constant<"1">), (Shadow $C)),
                    (FirstUse<"beta1"> (BlasCall<"lascl"> $layout, $uplo, ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, $beta, $n, $n, (Shadow $C), Alloca<1>))
                  )
                  >;

def spmv : CallBlasPattern<(Op $layout, $uplo, $n, $alpha, $ap, $x, $incx, $beta, $y, $incy),
                  ["y"],
                  [cblas_layout, uplo, len, fp, ap<["n"]>, vinc<["n"]>, fp, vinc<["n"]>],
                  [
                  /* alpha */ (Seq<["y0", "triangular", "n"], [], 1>
                    (BlasCall<"spmv"> $layout, $uplo, $n, Constant<"1.0">, $ap, $x, Constant<"0.0">, use<"y0">, ConstantInt<1>),
                    (BlasCall<"dot"> $n, (Shadow $y), use<"y0">, ConstantInt<1>)),
                  /* ap    */ (Seq<[], [], 1>
                    (BlasCall<"spr2"> $layout, $uplo, $n, $alpha, $x, (Shadow $y), (Shadow $ap)),
                    (DiagUpdateSPMV<""> $uplo, $n, $alpha, $x, (Shadow $y), (Shadow $ap))),
                  /* x     */ (BlasCall<"spmv"> $layout, $uplo, $n, $alpha, $ap, (Shadow $y), Constant<"1.0">, (Shadow $x)),
                  /* beta  */ (BlasCall<"dot"> $n, (Shadow $y), input<"y">),
                  /* y     */ (BlasCall<"scal"> $n, $beta, (Shadow $y))
                  ]
                  >;

def flip_uplo : IntMatchers<
    "charType", "charType",
    ["'l'", "'L'", "'U'", "'u'"],
    ["'u'", "'U'", "'L'", "'l'"]
    >;
// B2 = inv(A^T) B
// dB = inv(A^T) dB2
// d(A^T) −= dB B2^T
def trtrs : CallBlasPattern<(Op $layout, $uplo, $trans, $diag, $n, $nrhs, $A, $lda, $B, $ldb, $info),
                  ["B"],
                  [cblas_layout, uplo, trans, diag, len, len, mld<["uplo", "n", "n"]>, mld<["n", "nrhs"]>, info],
                  [
                  /* a     */ 
                    (Seq<["tri", "triangular", "n"], [], 1>
                      (BlasCall<"lacpy"> $layout, $uplo, $n, $n, (Shadow $A), use<"tri">, $n),
                      (BlasCall<"gemm">
                        $layout,
                        Char<"N">,
                        Char<"C">,
                        $n,
                        $n,
                        $nrhs,
                        Constant<"-1">,
                        (Rows $trans,
                          (Shadow $B),
                          (Concat $B, (ld $B, Char<"N">, $ldb, $n, $n))),
                        (Rows $trans,
                          (Concat $B, (ld $B, Char<"N">, $ldb, $n, $n)),
                          (Shadow $B)),
                        Constant<"1">,
                        use<"tri">, $n),

                      (BlasCall<"copy"> 
                          (ISelect (is_nonunit $diag), ConstantInt<0>, $n),
                          (First (Shadow $A)), (Add $lda, ConstantInt<1>),
                          use<"tri">, (Add $n, ConstantInt<1>)),

                      (BlasCall<"lacpy"> $layout, $uplo, $n, $n, use<"tri">, $n, (Shadow $A))
                      ),
                  /* b     */ (BlasCall<"trtrs"> $layout, $uplo, transpose<"trans">, $diag, $n, $nrhs, $A, (ld $A, Char<"N">, $lda, $n, $n), (Shadow $B), Alloca<1>),
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
// 
// 

//  Solve op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
def trsm: CallBlasPattern<(Op $layout, $side, $uplo, $transa, $diag, $m, $n, $alpha, $A, $lda, $B, $ldb),
                  ["B"],
                  [cblas_layout, side, uplo, trans, diag, len, len, fp, mld<["side", "m", "n"]>, mld<["m","n"]>],
                  [
                    /* alpha */ (AssertingInactiveArg),
                    /* A     */ (AssertingInactiveArg),
                    /* B     */ (AssertingInactiveArg),
                  ]
                  >;

def uplo_to_normal : IntMatchers<
    "charType", "charType",
    ["'l'", "'L'", "'U'", "'u'"],
    ["'n'", "'N'", "'T'", "'t'"]
    >;
def uplo_to_trans : IntMatchers<
    "charType", "charType",
    ["'l'", "'L'", "'U'", "'u'"],
    ["'t'", "'T'", "'N'", "'n'"]
    >;
def uplo_to_side : IntMatchers<
    "charType", "charType",
    ["'l'", "'L'", "'U'", "'u'"],
    ["'L'", "'L'", "'R'", "'R'"]
    >;
def uplo_to_rside : IntMatchers<
    "charType", "charType",
    ["'l'", "'L'", "'U'", "'u'"],
    ["'R'", "'R'", "'L'", "'L'"]
    >;


def potrf: CallBlasPattern<(Op $layout, $uplo, $n, $A, $lda, $info),
                  ["A"],
                  [cblas_layout, uplo, len, mld<["uplo", "n", "n"]>, info],
                  [
                    /* A     */ 
                  (Seq<["ztri", "zerotriangular", "n"], [], 1>
                      (BlasCall<"lacpy"> $layout, $uplo, $n, $n, (Shadow $A), use<"ztri">, $n),
                      
                      (BlasCall<"trmm"> $layout, (uplo_to_side $uplo), $uplo, Char<"T">, Char<"N">, $n, $n, Constant<"1.0">, $A, (ld $A, Char<"N">, $lda, $n, $n), use<"ztri">, $n),

                      (Seq<["tmp", "vector", "n"], [], 1>
                        
                        // Zero out flipped side again
                        (BlasCall<"copy"> $n, use<"ztri">, (Add $n, ConstantInt<1>), use<"tmp">, ConstantInt<1>),
                        (BlasCall<"scal"> $n, Constant<"0.5">, use<"tmp">, ConstantInt<1>),

                        (BlasCall<"lascl"> $layout, (flip_uplo $uplo), ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, Constant<"0.0">, $n, $n, use<"ztri">, $n, Alloca<1>),
                        (BlasCall<"copy"> $n, use<"tmp">, ConstantInt<1>, use<"ztri">, (Add $n, ConstantInt<1>))
                      ),
                      
                      (BlasCall<"trsm"> $layout, (uplo_to_rside $uplo), $uplo, Char<"N">, Char<"N">, $n, $n, Constant<"1.0">, $A, (ld $A, Char<"N">, $lda, $n, $n), use<"ztri">, $n),
                      
                      (BlasCall<"trsm"> $layout, (uplo_to_side $uplo), $uplo, Char<"T">, Char<"N">, $n, $n, Constant<"1.0">, $A, (ld $A, Char<"N">, $lda, $n, $n), use<"ztri">, $n) ,

                      (For<"i", 0> (Sub $n, ConstantInt<1>),
                        (BlasCall<"axpy">
                            (Sub (Sub $n, ConstantInt<1>), $i),
                            Constant<"1.0">,
                            (First
                                (Lookup $layout,
                                    (Concat use<"ztri">, $n),
                                    (ISelect (is_lower $uplo), $i, (Add $i, ConstantInt<1>)),
                                    (ISelect (is_lower $uplo), (Add $i, ConstantInt<1>), $i)
                                )
                            ),
                            (First
                                (Lookup $layout,
                                    (Concat ConstantInt<0>, $n),
                                    (ISelect (is_lower $uplo), ConstantInt<0>, ConstantInt<1>),
                                    (ISelect (is_lower $uplo), ConstantInt<1>, ConstantInt<0>)
                                )
                            ),
                            (First
                                (Lookup $layout,
                                    (Concat use<"ztri">, $n),
                                    (ISelect (is_lower $uplo), (Add $i, ConstantInt<1>), $i),
                                    (ISelect (is_lower $uplo), $i, (Add $i, ConstantInt<1>))
                                )
                            ),
                            (First
                                (Lookup $layout,
                                    (Concat ConstantInt<0>, $n),
                                    (ISelect (is_lower $uplo), ConstantInt<1>, ConstantInt<0>),
                                    (ISelect (is_lower $uplo), ConstantInt<0>, ConstantInt<1>)
                                )
                            )
                        )
                      ),
                     (BlasCall<"lacpy"> $layout, $uplo, $n, $n, use<"ztri">, $n, (Shadow $A))
                  )
                  ],
                  (Seq<["tri", "triangular", "n"], [], 0>
                      (BlasCall<"lacpy"> $layout, (flip_uplo $uplo), $n, $n, (Shadow $A), use<"tri">, $n),
                      
                      (CopyLowerToUpper<""> $layout, $uplo, (Shadow $A), $n),
                      
                      // Actual Math
                      (BlasCall<"trsm"> $layout, Char<"L">, $uplo, (uplo_to_normal $uplo), Char<"N">, $n, $n, Constant<"1.0">, $A, $lda, (Shadow $A)),
                      (BlasCall<"trsm"> $layout, Char<"R">, $uplo, (uplo_to_trans $uplo), Char<"N">, $n, $n, Constant<"1.0">, $A, $lda, (Shadow $A)),
                      (BlasCall<"scal"> $n, Constant<"0.5">, (First (Shadow $A)), (Add $lda, ConstantInt<1>)),

                     (Seq<["tmp", "vector", "n"], [], 1>

                        // Zero out flipped side again
                        (BlasCall<"copy"> $n, (First (Shadow $A)), (Add $lda, ConstantInt<1>), use<"tmp">, ConstantInt<1>),

                        (BlasCall<"lascl"> $layout, (flip_uplo $uplo), ConstantInt<0>, ConstantInt<0>, Constant<"1.0">, Constant<"0.0">, $n, $n, (Shadow $A), Alloca<1>),
                        (BlasCall<"copy"> $n, use<"tmp">, ConstantInt<1>, (First (Shadow $A)), (Add $lda, ConstantInt<1>)),

                        // More math
                        (BlasCall<"trmm"> $layout, (uplo_to_side $uplo), $uplo, Char<"N">, Char<"N">, $n, $n, Constant<"1.0">, $A, $lda, (Shadow $A)),    
                        // Restore the flipped side, but saving our computed diagonal
                        (BlasCall<"copy"> $n, (First (Shadow $A)), (Add $lda, ConstantInt<1>), use<"tmp">, ConstantInt<1>),
                        (BlasCall<"lacpy"> $layout, (flip_uplo $uplo), $n, $n, use<"tri">, $n, (Shadow $A)),
                        (BlasCall<"copy"> $n, use<"tmp">, ConstantInt<1>, (First (Shadow $A)), (Add $lda, ConstantInt<1>))
                    )
                  )
                  >;


def potrs: CallBlasPattern<(Op $layout, $uplo, $n, $nrhs, $A, $lda, $B, $ldb, $info),
                  ["B"],
                  [cblas_layout, uplo, len, len, mld<["uplo", "n", "n"]>, mld<["n", "nrhs"]>, info],
                  [
                    (Seq<["tri", "triangular", "n"], [], 1>

                      (BlasCall<"syr2k"> $layout, Char<"U">, Char<"N">, $n, $nrhs, Constant<"1.0">, input<"B">, $n, (Shadow $B), Constant<"0.0">, use<"tri">, $n),
                      (CopyLowerToUpper<""> $layout, Char<"U">, (Concat use<"tri">, $n), $n),

                      (BlasCall<"trsm"> $layout, (uplo_to_rside $uplo), $uplo, Char<"T">, Char<"N">, $n, $n, Constant<"1.0">, $A, (ld $A, Char<"N">, $lda, $n, $n), use<"tri">, $n),
                      
                      (BlasCall<"trsm"> $layout, (uplo_to_side $uplo), $uplo, Char<"N">, Char<"N">, $n, $n, Constant<"1.0">, $A, (ld $A, Char<"N">, $lda, $n, $n), use<"tri">, $n),
                      
                      (BlasCall<"trsm"> $layout, (uplo_to_side $uplo), $uplo, Char<"T">, Char<"N">, $n, $n, Constant<"1.0">, $A, (ld $A, Char<"N">, $lda, $n, $n), use<"tri">, $n),


                      (For<"i", 0> $n,
                        (BlasCall<"axpy">
                            (Sub $n, $i),
                            Constant<"-1.0">,
                            (First
                                (Lookup $layout,
                                    (Concat use<"tri">, $n),
                                    $i,
                                    $i
                                )
                            ),
                            (First
                                (Lookup $layout,
                                    (Concat ConstantInt<0>, $n),
                                    (ISelect (is_lower $uplo), ConstantInt<1>, ConstantInt<0>),
                                    (ISelect (is_lower $uplo), ConstantInt<0>, ConstantInt<1>)
                                )
                            ),
                            (First
                                (Lookup $layout,
                                    (Shadow $A),
                                    $i,
                                    $i
                                )
                            ),
                            (First
                                (Lookup $layout,
                                    (Concat ConstantInt<0>, $lda),
                                    (ISelect (is_lower $uplo), ConstantInt<1>, ConstantInt<0>),
                                    (ISelect (is_lower $uplo), ConstantInt<0>, ConstantInt<1>)
                                )
                            )
                        )
                     )
                    ),
                    (BlasCall<"potrs"> $layout, $uplo, $n, $nrhs, $A, (ld $A, Char<"N">, $lda, $n, $n), (Shadow $B), Alloca<1>)
                  ]
                  >;
