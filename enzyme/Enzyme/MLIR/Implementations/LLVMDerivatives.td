include "Common.td"

def : MemoryIdentityOp<"LLVM", "StoreOp", [1], [0]>;
def : InactiveOp<"LLVM", "SIToFPOp">;
def : InactiveOp<"LLVM", "UIToFPOp">;
def : InactiveOp<"LLVM", "FPToSIOp">;
def : InactiveOp<"LLVM", "FPToUIOp">;
def : InactiveOp<"LLVM", "AssumeOp">;
def : InactiveOp<"LLVM", "StackSaveOp">;
def : InactiveOp<"LLVM", "StackRestoreOp">;
def : InactiveOp<"LLVM", "NoAliasScopeDeclOp">;
def : InactiveOp<"LLVM", "LifetimeStartOp">;
def : InactiveOp<"LLVM", "LifetimeEndOp">;
def : InactiveOp<"LLVM", "Prefetch">;
def : InactiveOp<"LLVM", "MemsetOp">;

def : InactiveOp<"LLVM", "UndefOp">;
def : InactiveOp<"LLVM", "ConstantOp">;
def : InactiveOp<"LLVM", "UnreachableOp">;


def : ReadOnlyIdentityOp<"LLVM", "LoadOp", [0]>;
def : ReadOnlyIdentityOp<"LLVM", "AddrSpaceCastOp", [0]>;
def : ReadOnlyIdentityOp<"LLVM", "BitcastOp", [0]>;
def : ReadOnlyIdentityOp<"LLVM", "GEPOp", [0]>;
def : ReadOnlyIdentityOp<"LLVM", "PtrToIntOp", [0]>;
def : ReadOnlyIdentityOp<"LLVM", "IntToPtrOp", [0]>;

def : AllocationOp<"LLVM", "AllocaOp">;
