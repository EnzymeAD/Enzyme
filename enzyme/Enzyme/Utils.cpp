//===- Utils.cpp - Definition of miscellaneous utilities ------------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file defines miscellaneous utilities that are used as part of the
// AD process.
//
//===----------------------------------------------------------------------===//
#include "Utils.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace llvm;

EnzymeFailure::EnzymeFailure(llvm::StringRef RemarkName,
                             const llvm::DiagnosticLocation &Loc,
                             const llvm::Instruction *CodeRegion)
    : DiagnosticInfoIROptimization(
          EnzymeFailure::ID(), DS_Error, "enzyme", RemarkName,
          *CodeRegion->getParent()->getParent(), Loc, CodeRegion) {}

llvm::DiagnosticKind EnzymeFailure::ID() {
  static auto id = llvm::getNextAvailablePluginDiagnosticKind();
  return (llvm::DiagnosticKind)id;
}

/// \see DiagnosticInfoOptimizationBase::isEnabled.
bool EnzymeFailure::isEnabled() const { return true; }

/// Convert a floating type to a string
static inline std::string tofltstr(Type *T) {
  switch (T->getTypeID()) {
  case Type::HalfTyID:
    return "half";
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::X86_FP80TyID:
    return "x87d";
  case Type::FP128TyID:
    return "quad";
  case Type::PPC_FP128TyID:
    return "ppcddouble";
  default:
    llvm_unreachable("Invalid floating type");
  }
}

/// Create function for type that is equivalent to memcpy but adds to
/// destination rather than a direct copy; dst, src, numelems
Function *getOrInsertDifferentialFloatMemcpy(Module &M, PointerType *T,
                                             unsigned dstalign,
                                             unsigned srcalign) {
  Type *elementType = T->getElementType();
  assert(elementType->isFloatingPointTy());
  std::string name = "__enzyme_memcpyadd_" + tofltstr(elementType) + "da" +
                     std::to_string(dstalign) + "sa" + std::to_string(srcalign);
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()),
                        {T, T, Type::getInt64Ty(M.getContext())}, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::NoUnwind);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::NoCapture);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto dst = F->arg_begin();
  dst->setName("dst");
  auto src = dst + 1;
  src->setName("src");
  auto num = src + 1;
  num->setName("num");

  {
    IRBuilder<> B(entry);
    B.CreateCondBr(B.CreateICmpEQ(num, ConstantInt::get(num->getType(), 0)),
                   end, body);
  }

  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(num->getType(), 2, "idx");
    idx->addIncoming(ConstantInt::get(num->getType(), 0), entry);

    Value *dsti = B.CreateGEP(dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(dsti, "dst.i.l");
    StoreInst *dsts = B.CreateStore(Constant::getNullValue(elementType), dsti);
    if (dstalign) {
#if LLVM_VERSION_MAJOR >= 10
      dstl->setAlignment(Align(dstalign));
      dsts->setAlignment(Align(dstalign));
#else
      dstl->setAlignment(dstalign);
      dsts->setAlignment(dstalign);
#endif
    }

    Value *srci = B.CreateGEP(src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(srci, "src.i.l");
    StoreInst *srcs = B.CreateStore(B.CreateFAdd(srcl, dstl), srci);
    if (srcalign) {
#if LLVM_VERSION_MAJOR >= 10
      srcl->setAlignment(Align(srcalign));
      srcs->setAlignment(Align(srcalign));
#else
      srcl->setAlignment(srcalign);
      srcs->setAlignment(srcalign);
#endif
    }

    Value *next =
        B.CreateNUWAdd(idx, ConstantInt::get(num->getType(), 1), "idx.next");
    idx->addIncoming(next, body);
    B.CreateCondBr(B.CreateICmpEQ(num, next), end, body);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }
  return F;
}

// TODO implement differential memmove
Function *getOrInsertDifferentialFloatMemmove(Module &M, PointerType *T,
                                              unsigned dstalign,
                                              unsigned srcalign) {
  llvm::errs() << "warning: didn't implement memmove, using memcpy as fallback "
                  "which can result in errors\n";
  return getOrInsertDifferentialFloatMemcpy(M, T, dstalign, srcalign);
}

/// Create function to computer nearest power of two
llvm::Value *nextPowerOfTwo(llvm::IRBuilder<> &B, llvm::Value *V) {
  assert(V->getType()->isIntegerTy());
  IntegerType *T = cast<IntegerType>(V->getType());
  V = B.CreateAdd(V, ConstantInt::get(T, -1));
  for (size_t i = 1; i < T->getBitWidth(); i *= 2) {
    V = B.CreateOr(V, B.CreateLShr(V, ConstantInt::get(T, i)));
  }
  V = B.CreateAdd(V, ConstantInt::get(T, 1));
  return V;
}

llvm::Function *getOrInsertDifferentialMPI_Wait(llvm::Module &M,
                                                ArrayRef<llvm::Type *> T,
                                                Type *reqType) {
  std::vector<llvm::Type *> types(T.begin(), T.end());
  types.push_back(reqType);
  std::string name = "__enzyme_differential_mpi_wait";
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()), types, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *isend = BasicBlock::Create(M.getContext(), "invertISend", F);
  BasicBlock *irecv = BasicBlock::Create(M.getContext(), "invertIRecv", F);

#if 0
    /*0 */Type::getInt8PtrTy(call.getContext())
    /*1 */i64
    /*2 */Type::getInt8PtrTy(call.getContext())
    /*3 */i64
    /*4 */i64
    /*5 */Type::getInt8PtrTy(call.getContext())
    /*6 */Type::getInt8Ty(call.getContext())
#endif

  auto buf = F->arg_begin();
  buf->setName("buf");
  Value *count = buf + 1;
  count->setName("count");
  Value *datatype = buf + 2;
  datatype->setName("datatype");
  Value *source = buf + 3;
  source->setName("source");
  Value *tag = buf + 4;
  tag->setName("tag");
  Value *comm = buf + 5;
  comm->setName("comm");
  Value *fn = buf + 6;
  fn->setName("fn");
  Value *d_req = buf + 7;
  d_req->setName("d_req");

  auto isendfn = M.getFunction("MPI_Isend");
  assert(isendfn);
  auto irecvfn = M.getFunction("MPI_Irecv");
  assert(irecvfn);

  IRBuilder<> B(entry);
  auto arg = isendfn->arg_begin();
  arg++;
  count = B.CreateZExtOrTrunc(count, arg->getType());
  arg++;
  datatype = B.CreatePointerCast(datatype, arg->getType());
  arg++;
  source = B.CreateZExtOrTrunc(source, arg->getType());
  arg++;
  tag = B.CreateZExtOrTrunc(tag, arg->getType());
  arg++;
  comm = B.CreatePointerCast(comm, arg->getType());

  Value *args[] = {
      buf, count, datatype, source, tag, comm, d_req,
  };

  B.CreateCondBr(B.CreateICmpEQ(fn, ConstantInt::get(fn->getType(),
                                                     (int)MPI_CallType::ISEND)),
                 isend, irecv);

  {
    B.SetInsertPoint(isend);
    auto fcall = B.CreateCall(irecvfn, args);
    fcall->setCallingConv(isendfn->getCallingConv());
    B.CreateRetVoid();
  }

  {
    B.SetInsertPoint(irecv);
    auto fcall = B.CreateCall(isendfn, args);
    fcall->setCallingConv(isendfn->getCallingConv());
    B.CreateRetVoid();
  }
  return F;
}

llvm::Value *getOrInsertOpFloatSum(llvm::Module &M,
                                   llvm::Type* OpPtr,
                                   ConcreteType CT,
                                   llvm::Type* intType) {
  std::string name = "__enzyme_mpi_sum" + CT.str();
  assert(CT.isFloat()); 
  auto FT = CT.isFloat();
  return M.getOrInsertGlobal(name, cast<PointerType>(OpPtr)->getElementType(), [&]() -> GlobalVariable* {
    std::vector<llvm::Type *> types = {
                                        PointerType::getUnqual(FT),
                                        PointerType::getUnqual(FT),
                                        PointerType::getUnqual(intType),
                                        OpPtr
                                        };
    FunctionType *FT =
          FunctionType::get(Type::getVoidTy(M.getContext()), types, false);

    #if LLVM_VERSION_MAJOR >= 9
      Function *F = cast<Function>(M.getOrInsertFunction(name+"_run", FT).getCallee());
    #else
      Function *F = cast<Function>(M.getOrInsertFunction(name+"_run", FT));
    #endif

      F->setLinkage(Function::LinkageTypes::InternalLinkage);
      F->addFnAttr(Attribute::ArgMemOnly);
      F->addFnAttr(Attribute::NoUnwind);
      F->addParamAttr(0, Attribute::NoCapture);
      F->addParamAttr(0, Attribute::ReadOnly);
      F->addParamAttr(1, Attribute::NoCapture);
      F->addParamAttr(2, Attribute::NoCapture);
      F->addParamAttr(2, Attribute::ReadOnly);
      F->addParamAttr(3, Attribute::NoCapture);
      F->addParamAttr(3, Attribute::ReadNone);

      BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
      BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
      BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

      auto src = F->arg_begin();
      src->setName("src");
      auto dst = src + 1;
      dst->setName("dst");
      auto lenp = src + 1;
      lenp->setName("lenp");
      Value *len;
      // TODO consider using datatype arg and asserting same size as assumed by type analysis

      {
        IRBuilder<> B(entry);
        len = B.CreateLoad(lenp);
        B.CreateCondBr(B.CreateICmpEQ(len, ConstantInt::get(len->getType(), 0)),
                        end, body);
      }

      {
        IRBuilder<> B(body);
        B.setFastMathFlags(getFast());
        PHINode *idx = B.CreatePHI(len->getType(), 2, "idx");
        idx->addIncoming(ConstantInt::get(len->getType(), 0), entry);

        Value *dsti = B.CreateGEP(dst, idx, "dst.i");
        LoadInst *dstl = B.CreateLoad(dsti, "dst.i.l");

        Value *srci = B.CreateGEP(src, idx, "src.i");
        LoadInst *srcl = B.CreateLoad(srci, "src.i.l");

        B.CreateStore(B.CreateFAdd(srcl, dstl), dsti);

        Value *next =
            B.CreateNUWAdd(idx, ConstantInt::get(len->getType(), 1), "idx.next");
        idx->addIncoming(next, body);
        B.CreateCondBr(B.CreateICmpEQ(len, next), end, body);
      }

      {
        IRBuilder<> B(end);
        B.CreateRetVoid();
      }


    std::vector<llvm::Type *> rtypes = {
                                        Type::getInt8PtrTy(M.getContext()),
                                        intType,
                                        OpPtr
                                        };
    FunctionType *RFT =
          FunctionType::get(intType, rtypes, false);

    #if LLVM_VERSION_MAJOR >= 9
      Function *RF = cast<Function>(M.getOrInsertFunction("MPI_Op_create", RFT).getCallee());
    #else
      Function *RF = cast<Function>(M.getOrInsertFunction("MPI_Op_create", RFT));
    #endif

      GlobalVariable* GV = new GlobalVariable(M, cast<PointerType>(OpPtr)->getElementType(), false, GlobalVariable::InternalLinkage,
                               nullptr, name);
                               


    // Finish initializing mpi sum https://www.mpich.org/static/docs/v3.2/www3/MPI_Op_create.html
    FunctionType *IFT =
          FunctionType::get(Type::getVoidTy(M.getContext()), ArrayRef<Type*>(), false);

    #if LLVM_VERSION_MAJOR >= 9
      Function *initializerFunction = cast<Function>(M.getOrInsertFunction("MPI_Sum_initializer", IFT).getCallee());
    #else
      Function *initializerFunction = cast<Function>(M.getOrInsertFunction("MPI_Sum_initializer", IFT));
    #endif
    
    initializerFunction->setLinkage(Function::LinkageTypes::InternalLinkage);
    initializerFunction->addFnAttr(Attribute::NoUnwind);

    {
    BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", initializerFunction);
    IRBuilder <> B(entry);
    Value *args[] = { ConstantExpr::getPointerCast(F, rtypes[0]), ConstantInt::get(rtypes[1], 1, false), ConstantExpr::getPointerCast(GV, rtypes[2])};
    B.CreateCall(RF, args);
    B.CreateRetVoid();
    }

    // https://llvm.org/docs/LangRef.html#the-llvm-global-ctors-global-variable
      GlobalVariable *ctors = M.getGlobalVariable("llvm.global_ctors");
      
      SmallVector<Constant *, 10> CAList;
      if (ctors) {
        ConstantArray *OldCA = cast<ConstantArray>(ctors->getInitializer());
        for (unsigned I = 0, E = OldCA->getNumOperands(); I < E; ++I)
          CAList.push_back(OldCA->getOperand(I));
      }
      Constant* newConstructor[] = {ConstantInt::get(Type::getInt32Ty(M.getContext()), 65535, false), initializerFunction, ConstantExpr::getPointerCast(GV, Type::getInt8PtrTy(M.getContext()))};
      CAList.push_back(ConstantStruct::getAnon(M.getContext(), newConstructor));
      ArrayType *ATy = ArrayType::get(CAList[0]->getType(), CAList.size());
      Constant *CA = ConstantArray::get(ATy, CAList);
      
      {
        // Create the new global and insert it next to the existing list.
        bool isConstant = ctors ? ctors->isConstant() : true;
        auto linkage = ctors ? ctors->getLinkage() : GlobalVariable::AppendingLinkage;

        GlobalVariable *NGV =
            new GlobalVariable(M, CA->getType(), isConstant, linkage, CA, "");

        if (ctors)
          NGV->takeName(ctors);
        else
          NGV->setName("llvm.global_ctors");
        
        // Nuke the old list, replacing any uses with the new one.
        if (ctors && !ctors->use_empty()) {
          Constant *V = ConstantExpr::getBitCast(NGV, ctors->getType());
          ctors->replaceAllUsesWith(V);
        }
        ctors->eraseFromParent();
      }

      return GV;
  });
}
