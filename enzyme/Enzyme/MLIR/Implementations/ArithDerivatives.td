class MLIRDerivative<string dialect_, string opName_, dag patternToMatch, list<dag> resultOps> {
  string dialect = dialect_;
  string opName = opName_;
  dag PatternToMatch = patternToMatch;
  list<dag> ArgDerivatives = resultOps;
}

class Operation<bit usesPrimal_, bit usesShadow_, bit usesCustom_=0> {
  bit usesPrimal = usesPrimal_;
  bit usesShadow = usesShadow_;
  bit usesCustom = usesCustom_;
}

class DiffeRetIndex<list<int> indices_> {
  list<int> indices = indices_;
}
def DiffeRet : DiffeRetIndex<[-1]>;

class Inst<string mnemonic, string dialect_>  : Operation</*primal*/1, /*shadow*/0> {
  string name = mnemonic;
  string dialect = dialect_;
}
class ArithInst<string m> : Inst<m, "arith">;

def AddF : ArithInst<"arith::AddFOp">;
def SubF : ArithInst<"arith::SubFOp">;
def NegF : ArithInst<"arith::NegFOp">;
def MulF : ArithInst<"arith::MulFOp">;
def DivF : ArithInst<"arith::DivFOp">;
def RemF : ArithInst<"arith::RemFOp">;

def CheckedMulF : ArithInst<"arith::MulFOp">;
def CheckedDivF : ArithInst<"arith::DivFOp">;

def Op {
}

def : MLIRDerivative<"arith", "AddFOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ]
                  >;

def : MLIRDerivative<"arith", "SubFOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (NegF (DiffeRet)),
                    ]
                  >;
def : MLIRDerivative<"arith", "NegFOp", (Op $x),
                    [
                      (NegF (DiffeRet)),
                    ]
                  >;
def : MLIRDerivative<"arith", "MulFOp", (Op $x, $y),
                    [
                      (CheckedMulF (DiffeRet), $y),
                      (CheckedMulF (DiffeRet), $x)
                    ]
                  >;
def : MLIRDerivative<"arith", "DivFOp", (Op $x, $y), 
                    [
                      (CheckedDivF (DiffeRet), $y),
                      (NegF (MulF (CheckedDivF (DiffeRet), $y), (DivF $x, $y)))
                    ]
                    // (CheckedDiv (FSub (SelectIfActive $x, (FMul (Shadow $x), $y), (Zero $x)), (SelectIfActive $y, (FMul (Shadow $y), $x), (Zero $y))), (FMul $y, $y))
                  >;
