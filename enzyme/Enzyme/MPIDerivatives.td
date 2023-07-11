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
def MemCopyFloats: MagicInst;
def MPISum: MagicInst;
def MemSet: MagicInst;
def False: MagicInst;
def True: MagicInst;
def MPITySize: MagicInst;

// as CallBlasPattern
class CallMPIPattern<dag patternToMatch, list<string> mutables, list<MPI_ArgType> paramTypes, 
                      list<dag> resultOps> {
  dag PatternToMatch = patternToMatch;
  list<MPI_ArgType> inputTypes = paramTypes;
  list<dag> ArgDerivatives = resultOps;
  list<string> mutable = mutables;
}

// def MPI_Isend : CallMPIPattern<(Op $buf, $count, $datatype, $source, $tag, $comm, $request),
//                     ["request"], [buf, size, datatype, integer, integer, comm, request],
//                     [

//                     ]>;

def MPI_Gather : CallMPIPattern<(Op $sendbuf, $sendcount, $sendtype, $recvbuf, $recvcount, $recvtype, $root, $comm),
                    ["recvbuf"], [buf, size, datatype, buf, size, datatype, integer, comm],
                    [
                      (b<"MPI_Scatter"> adj<"recvbuf">, $recvcount, $recvtype, $buf, $sendcount, $sendtype, $root, $comm),
                      // TODO: if root, Zero diff(recvbuffer) [memset to 0]
                      (MemCopyFloats $buf, adj<"sendbuf">, (Mul $sendcount, (MPITySize $sendtype))),
                    ]>;

def MPI_AllGather : CallMPIPattern<(Op $sendbuf, $sendcount, $sendtype, $recvbuf, $recvcount, $recvtype, $comm),
                    ["recvbuf"], [buf, size, datatype, buf, size, datatype, comm],
                    [
                      (b<"MPI_Reduce_scatter_block"> adj<"recvbuf">, $buf, $sendcount, $sendtype, MPISum, $comm),
                      (MemSet adj<"recvbuf">, 0, (Mul $recvcount, (MPITySize $recvtype)), False),
                      (MemCopyFloats $buf, adj<"sendbuf">, (Mul $sendcount, (MPITySize $sendtype))),
                    ]>;
