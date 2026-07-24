include "Common.td"

// Memory / pointer ops of the FIR and HLFIR dialects, treated as active-memory
// identities so Enzyme can differentiate through by-reference Fortran variables
// (real Fortran arguments are !fir.ref<...>). Mirrors MemRefDerivatives.td.
//
// The pointer operand indices below are 0-based; storedvals mark which operand
// is the value being written (for the store-like ops).

// Scope marker carrying no differentiable data.
def : InactiveOp<"fir", "DummyScopeOp">;

// Stack/heap allocations: the shadow gets a parallel allocation.
def : AllocationOp<"fir", "AllocaOp">;
def : AllocationOp<"fir", "AllocMemOp">;

// Loads read through a pointer (operand 0); the tangent loads from the shadow.
def : ReadOnlyIdentityOp<"fir", "LoadOp", [0]>;

// Stores write operand 0 through pointer operand 1; the tangent stores the
// value's tangent through the shadow pointer.
def : MemoryIdentityOp<"fir", "StoreOp", [1], [0]>;

// hlfir.declare re-derives variable/box handles from a pointer (operand 0);
// the shadow declares the shadow pointer. Both results follow.
def : ReadOnlyIdentityOp<"hlfir", "DeclareOp", [0]>;

// hlfir.assign writes operand 0 (the RHS value) through operand 1 (the LHS
// variable); behaves like a store for the active-memory model.
def : MemoryIdentityOp<"hlfir", "AssignOp", [1], [0]>;
