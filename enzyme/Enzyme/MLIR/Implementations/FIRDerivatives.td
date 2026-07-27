include "Common.td"

// FIR/HLFIR memory ops as active-memory identities, so Enzyme can differentiate
// through by-reference Fortran variables (!fir.ref<...>). Mirrors
// MemRefDerivatives.td: the lists hold 0-based pointer operand indices, then the
// stored-value indices for store-like ops.

def : InactiveOp<"fir", "DummyScopeOp">;

def : AllocationOp<"fir", "AllocaOp">;
def : AllocationOp<"fir", "AllocMemOp">;

def : ReadOnlyIdentityOp<"fir", "LoadOp", [0]>;
def : MemoryIdentityOp<"fir", "StoreOp", [1], [0]>;

def : ReadOnlyIdentityOp<"hlfir", "DeclareOp", [0]>;
def : MemoryIdentityOp<"hlfir", "AssignOp", [1], [0]>;
