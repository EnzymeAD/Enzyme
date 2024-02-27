include "Common.td"

def : MemoryIdentityOp<"memref", "StoreOp", [1], [0]>;
def : ReadOnlyIdentityOp<"memref", "LoadOp", [0]>;
def : ReadOnlyIdentityOp<"memref", "CastOp", [0]>;
def : ReadOnlyIdentityOp<"memref", "CollapseShapeOp", [0]>;
def : ReadOnlyIdentityOp<"memref", "ExpandShapeOp", [0]>;
def : ReadOnlyIdentityOp<"memref", "ReinterpretCastOp", [0]>;
def : ReadOnlyIdentityOp<"memref", "ReshapeOp", [0]>;
def : ReadOnlyIdentityOp<"memref", "TransposeOp", [0]>;
def : ReadOnlyIdentityOp<"memref", "ViewOp", [0]>;
def : ReadOnlyIdentityOp<"memref", "SubViewOp", [0]>;

def : InactiveOp<"memref", "DimOp">;
def : AllocationOp<"memref", "AllocOp">;
def : AllocationOp<"memref", "AllocaOp">;
