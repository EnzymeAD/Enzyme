class MPI_CallType <int _id> {
  // value for MPI_Call
  int id = _id;
}

class MPI_Elem <int _id> {
    // offset of the struct field in getMPIHelper
    int id = _id;
}

// MPI_CallType
def MPI_ISEND: MPI_CallType<0>;
def MPI_IRECV: MPI_CallType<1>;

// MPI_Elem
def MPI_Buf: MPI_Elem<0>;
def MPI_Count: MPI_Elem<1>;
def MPI_DataType: MPI_Elem<2>;
def MPI_Src: MPI_Elem<3>;
def MPI_Tag: MPI_Elem<4>;
def MPI_Comm: MPI_Elem<5>;
def MPI_Call: MPI_Elem<6>;
def MPI_Old: MPI_Elem<7>;

class MPI_ArgType <int _id> {
  int id = _id;
}

// MPI_ArgType
def buf: MPI_ArgType<0>;
def size: MPI_ArgType<1>; 
def datatype: MPI_ArgType<2>;
def integer: MPI_ArgType<3>;
def comm: MPI_ArgType<4>;
def request: MPI_ArgType<5>;

// special op
def DiffMemCopyFloats: MagicInst;
def MPI_SUM: MagicInst;
def MemSet: MagicInst;
def False: MagicInst;
def True: MagicInst;
def MPITypeSize: MagicInst;
def MPICommRank: MagicInst;
def CreateAllocation: MagicInst;

class AutoTypesFunc<string name_> {
  string name = name_;
}

class CallMPIPattern<dag patternToMatch, list<string> funcNames, list<string> mutables,
                      list<MPI_ArgType> paramTypes, list<dag> resultOps, dag forwardOps>
       : CallPattern<patternToMatch, funcNames, resultOps, forwardOps> {
  list<MPI_ArgType> inputTypes = paramTypes;
  list<string> mutable = mutables;
}

// def MPI_Isend : CallMPIPattern<(Op $buf, $count, $datatype, $source, $tag, $comm, $request),
//                     ["MPI_Isend", "PMPI_Isend"], ["request"], 
//                     [buf, size, datatype, integer, integer, comm, request],
//                     [
//
//                     ],
//                     (ForwardFromSummedReverse)>;

def MPI_Gather : CallMPIPattern<(Op $sendbuf, $sendcount, $sendtype, $recvbuf, $recvcount, $recvtype, $root, $comm),
                    ["MPI_Gather"], ["recvbuf"], [buf, size, datatype, buf, size, datatype, integer, comm],
                    [
                      (CreateAllocation $buf, (Mul $sendcount, (MPITypeSize $sendtype)))
                      (Call (SameTypesFunc<"MPI_Scatter">), (Shadow $recvbuf), $recvcount, $recvtype, $buf, $sendcount, $sendtype, $root, $comm),
                      (Select (ICmpEQ $root (MPICommRank $comm)) 
                        (MemSet (Shadow $recvbuf), Zero, (Mul $recvcount, (MPITypeSize $recvtype)), False),
                        (Noop))
                      (DiffMemCopyFloats $sendbuf, $buf, (Shadow $sendbuf), (Mul $sendcount, (MPITypeSize $sendtype))),
                    ],
                    (ForwardFromSummedReverse)>;

def MPI_AllGather : CallMPIPattern<(Op $sendbuf, $sendcount, $sendtype, $recvbuf, $recvcount, $recvtype, $comm),
                    ["MPI_AllGather"], ["recvbuf"], [buf, size, datatype, buf, size, datatype, comm],
                    [
                      (CreateAllocation $buf, (Mul $sendcount, (MPITypeSize $sendtype)))
                      (Call (AutoTypesFunc<"MPI_Reduce_scatter_block">), (Shadow $recvbuf), $buf, $sendcount, $sendtype, MPI_SUM, $comm),
                      (MemSet (Shadow $recvbuf), Zero, (Mul $recvcount, (MPITypeSize $recvtype)), False),
                      (DiffMemCopyFloats $sendbuf, $buf, (Shadow $sendbuf), (Mul $sendcount, (MPITypeSize $sendtype))),
                    ],
                    (ForwardFromSummedReverse)>;

def MPI_Scatter   : CallMPIPattern<(Op $sendbuf, $sendcount, $sendtype, $recvbuf, $recvcount, $recvtype, $root, $comm),
                    ["MPI_Scatter"], ["recvbuf"], [buf, size, datatype, buf, size, datatype, integer, comm],
                    [
                      (Select (ICmpEQ $root (MPICommRank $comm)) 
                        (CreateAllocation $buf, (Mul $sendcount, (MPITypeSize $sendtype))),
                        (Assign $buf, UndefValue))
                      (Call SameTypesFunc<"MPI_Gather">, (Shadow $recvbuf), $buf, $recvcount, $recvtype, $sendcount, $sendtype, $root, $comm),
                      (MemSet (Shadow $recvbuf), Zero, (Mul $recvcount, (MPITypeSize $recvtype)), False),
                      (Select (ICmpEQ $root (MPICommRank $comm)) 
                        (DiffMemCopyFloats $sendbuf, $buf, (Shadow $sendbuf), (Mul $sendcount, (MPITypeSize $sendtype))),
                        (Noop)
                      )
                    ],
                    (ForwardFromSummedReverse)>;
