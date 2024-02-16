#ifndef ENZYME_MLIR_IMPLEMENTATIONS_COMMON
#define ENZYME_MLIR_IMPLEMENTATIONS_COMMON

class InactiveOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class AllocationOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class ControlFlowOp<string dialect_, string opName_, string impl_> {
  string dialect = dialect_;
  string opName = opName_;
  string impl = impl_;
}

class MemoryIdentityOp<string dialect_, string opName_, list<int> ptrargs_, list<int> storedargs_ = []> {
  string dialect = dialect_;
  string opName = opName_;
  list<int> ptrargs = ptrargs_;
  list<int> storedargs = storedargs_;
}

class ReadOnlyIdentityOp<string dialect_, string opName_, list<int> ptrargs_> : MemoryIdentityOp<dialect_, opName_, ptrargs_>;

class BranchOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

class RegionTerminatorOp<string dialect_, string opName_> {
  string dialect = dialect_;
  string opName = opName_;
}

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

def Op {
}

class ArithInst<string m> : Inst<m, "arith">;
class MathInst<string m> : Inst<m, "math">;

def AddF : ArithInst<"AddFOp">;
def SubF : ArithInst<"SubFOp">;
def NegF : ArithInst<"NegFOp">;
def MulF : ArithInst<"MulFOp">;
def DivF : ArithInst<"DivFOp">;
def RemF : ArithInst<"RemFOp">;

def CheckedMulF : ArithInst<"MulFOp">;
def CheckedDivF : ArithInst<"DivFOp">;

def CosF : MathInst<"CosOp">;
def SinF : MathInst<"SinOp">;
def ExpF : MathInst<"ExpOp">;
def SqrtF : MathInst<"SqrtOp">;

#endif // ENZYME_MLIR_IMPLEMENTATIONS_COMMON
