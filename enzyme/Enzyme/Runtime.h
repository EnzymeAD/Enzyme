#ifndef ENZYME_RUNTIME_H
#define ENZYME_RUNTIME_H

#include "llvm/ADT/ArrayRef.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Support/CommandLine.h"

enum class ErrorType {
  NoDerivative = 0,
  NoShadow = 1,
  IllegalTypeAnalysis = 2,
  NoType = 3,
  IllegalFirstPointer = 4,
  InternalError = 5,
  TypeDepthExceeded = 6
};

extern "C" {
/// Print additional debug info relevant to performance
extern void (*CustomErrorHandler)(const char *, LLVMValueRef, ErrorType,
                                  const void *);
}

/// Create function for type that performs the derivative memcpy on floating
/// point memory
llvm::Function *
getOrInsertDifferentialFloatMemcpy(llvm::Module &M, llvm::Type *T,
                                   unsigned dstalign, unsigned srcalign,
                                   unsigned dstaddr, unsigned srcaddr);

/// Create function for type that performs memcpy with a stride
llvm::Function *getOrInsertMemcpyStrided(llvm::Module &M, llvm::PointerType *T,
                                         llvm::Type *IT, unsigned dstalign,
                                         unsigned srcalign);

/// Create function for type that performs the derivative memmove on floating
/// point memory
llvm::Function *
getOrInsertDifferentialFloatMemmove(llvm::Module &M, llvm::Type *T,
                                    unsigned dstalign, unsigned srcalign,
                                    unsigned dstaddr, unsigned srcaddr);

llvm::Function *getOrInsertCheckedFree(llvm::Module &M, llvm::CallInst *call,
                                       llvm::Type *Type, unsigned width);

/// Create function for type that performs the derivative MPI_Wait
llvm::Function *getOrInsertDifferentialMPI_Wait(llvm::Module &M,
                                                llvm::ArrayRef<llvm::Type *> T,
                                                llvm::Type *reqType);

void ErrorIfRuntimeInactive(llvm::IRBuilder<> &B, llvm::Value *primal,
                            llvm::Value *shadow, const char *Message,
                            llvm::DebugLoc &&loc, llvm::Instruction *orig);

llvm::Function *
getOrInsertDifferentialWaitallSave(llvm::Module &M,
                                   llvm::ArrayRef<llvm::Type *> T,
                                   llvm::PointerType *reqType);

#endif
