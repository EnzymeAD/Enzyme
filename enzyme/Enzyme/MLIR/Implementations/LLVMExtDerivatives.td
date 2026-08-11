include "Common.td"

def : InactiveOp<"llvm_ext", "FreeOp">;
def : InactiveOp<"llvm_ext", "PtrSizeHintOp">;

def : AllocationOp<"llvm_ext", "AllocOp">;
