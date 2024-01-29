include "Common.td"

def : InactiveOp<"LLVM", "SIToFPOp">;
def : InactiveOp<"LLVM", "UIToFPOp">;
def : InactiveOp<"LLVM", "FPToSIOp">;
def : InactiveOp<"LLVM", "FPToUIOp">;
def : InactiveOp<"LLVM", "AssumeOp">;
def : InactiveOp<"LLVM", "StackSaveOp">;
def : InactiveOp<"LLVM", "StackRestoreOp">;
def : InactiveOp<"LLVM", "LifetimeStartOp">;
def : InactiveOp<"LLVM", "LifetimeEndOp">;
def : InactiveOp<"LLVM", "Prefetch">;
def : InactiveOp<"LLVM", "MemsetOp">;

def : InactiveOp<"LLVM", "UndefOp">;
def : InactiveOp<"LLVM", "ConstantOp">;
def : InactiveOp<"LLVM", "UnreachableOp">;
