//===- BlasGetrs.cpp - Derivatives of LAPACK LU solves
//---------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BlasGetrs.h"
#include <functional>

using namespace llvm;

namespace {
// Column-major helpers use pointer-sized offsets even for an LP64 BLAS call.
// In particular, lda * column must not overflow a 32-bit BLAS integer.
struct GetrsBuilder {
  Module &M;
  Function *Caller;
  const BlasInfo &Blas;
  Function *F;
  IRBuilder<> B;
  Type *FP;
  IntegerType *IT;
  IntegerType *Index;

  GetrsBuilder(Module &M, Function *Caller, const BlasInfo &Blas, Function *F)
      : M(M), Caller(Caller), Blas(Blas), F(F),
        B(BasicBlock::Create(M.getContext(), "entry", F)),
        FP(Blas.fpType(M.getContext())), IT(Blas.intType(M.getContext())),
        Index(M.getDataLayout().getIntPtrType(M.getContext())) {}

  Value *index(Value *V) { return B.CreateZExtOrTrunc(V, Index); }
  Value *ptr(Type *T, Value *V) {
    auto P = getUnqual(T);
    return V->getType()->isIntegerTy() ? B.CreateIntToPtr(V, P)
                                       : B.CreatePointerCast(V, P);
  }
  Value *at(Type *T, Value *P, Value *I) {
    return B.CreateInBoundsGEP(T, ptr(T, P), index(I));
  }
  Value *at(Value *P, Value *Row, Value *Col, Value *LD) {
    auto Offset = B.CreateAdd(index(Row), B.CreateMul(index(Col), index(LD)));
    return at(FP, P, Offset);
  }
  Value *load(Value *P, Value *Row, Value *Col, Value *LD) {
    return B.CreateLoad(FP, at(P, Row, Col, LD));
  }
  void store(Value *V, Value *P, Value *Row, Value *Col, Value *LD) {
    B.CreateStore(V, at(P, Row, Col, LD));
  }
  void loop(Value *Count, const Twine &Name,
            const std::function<void(Value *)> &Body) {
    auto Pre = B.GetInsertBlock();
    auto BB = BasicBlock::Create(M.getContext(), Name, F);
    auto End = BasicBlock::Create(M.getContext(), Name + ".end", F);
    B.CreateCondBr(
        B.CreateICmpSGT(Count, ConstantInt::get(Count->getType(), 0)), BB, End);
    B.SetInsertPoint(BB);
    auto I = B.CreatePHI(Count->getType(), 2, Name + ".i");
    I->addIncoming(ConstantInt::get(Count->getType(), 0), Pre);
    Body(I);
    auto Next = B.CreateAdd(I, ConstantInt::get(Count->getType(), 1));
    auto Latch = B.GetInsertBlock();
    B.CreateCondBr(B.CreateICmpULT(Next, Count), BB, End);
    I->addIncoming(Next, Latch);
    B.SetInsertPoint(End);
  }
  Value *allocate(Type *T, Value *Rows, Value *Cols, const Twine &Name) {
    return ptr(
        T, CreateAllocation(B, T, B.CreateMul(index(Rows), index(Cols)), Name));
  }
  void copy(Value *Dst, Value *Src, Value *Rows, Value *Cols, Value *SrcLD) {
    loop(Cols, "copy.col", [&](Value *J) {
      loop(Rows, "copy.row",
           [&](Value *I) { store(load(Src, I, J, SrcLD), Dst, I, J, Rows); });
    });
  }
  Value *ref(Value *V) {
    IRBuilder<> Entry(&F->getEntryBlock(), F->getEntryBlock().begin());
    auto A = Entry.CreateAlloca(V->getType());
    B.CreateStore(V, A);
    return A;
  }
  Value *character(char C) { return ref(B.getInt8(C)); }
  Value *integer(int V) { return ref(ConstantInt::get(IT, V, true)); }
  Value *real(double V) { return ref(ConstantFP::get(FP, V)); }

  // Use the original library's symbol spelling, including ILP64 and Julia
  // library selectors. Existing declarations determine their pointer spelling.
  void call(StringRef Op, SmallVector<Value *, 16> Args, unsigned NChars = 0) {
    std::string Name = Blas.prefix + Blas.floatType + Op.str() + Blas.suffix;
    std::string Renamed = getRenamedPerCallingConv(Caller->getName(), Name);
    Function *Target = M.getFunction(Renamed);
    if (!Target) {
      // These are Fortran entry points. New declarations include the hidden
      // CHARACTER lengths; an existing declaration retains its exact ABI.
      for (unsigned I = 0; I < NChars; ++I)
        Args.push_back(ConstantInt::get(Index, 1));
      SmallVector<Type *, 16> Types;
      for (Value *V : Args)
        Types.push_back(V->getType());
      Target = cast<Function>(
          M.getOrInsertFunction(Renamed,
                                FunctionType::get(B.getVoidTy(), Types, false))
              .getCallee());
    }
    auto FT = Target->getFunctionType();
    while (Args.size() < FT->getNumParams())
      Args.push_back(ConstantInt::get(FT->getParamType(Args.size()), 1));
    assert(Args.size() == FT->getNumParams() && "unexpected LAPACK/BLAS ABI");
    for (unsigned I = 0; I < Args.size(); ++I) {
      Type *T = FT->getParamType(I);
      if (Args[I]->getType() == T)
        continue;
      if (T->isIntegerTy() && Args[I]->getType()->isPointerTy())
        Args[I] = B.CreatePtrToInt(Args[I], T);
      else if (T->isPointerTy() && Args[I]->getType()->isIntegerTy())
        Args[I] = B.CreateIntToPtr(Args[I], T);
      else
        Args[I] = B.CreatePointerCast(Args[I], T);
    }
    auto C = B.CreateCall(Target, Args);
    C->setCallingConv(Target->getCallingConv());
  }
  void solve(Value *Trans, Value *N, Value *NRHS, Value *A, Value *LDA,
             Value *Piv, Value *RHS, Value *LDB) {
    auto Info = integer(0);
    call("getrs",
         {ref(Trans), ref(N), ref(NRHS), A, ref(LDA), Piv, RHS, ref(LDB), Info},
         1);
  }
  void trmm(char Uplo, char Trans, char Diag, Value *N, Value *NRHS, Value *A,
            Value *LDA, Value *RHS) {
    call("trmm",
         {character('L'), character(Uplo), character(Trans), character(Diag),
          ref(N), ref(NRHS), real(1), A, ref(LDA), RHS, ref(N)},
         4);
  }
  void permute(Value *X, Value *N, Value *NRHS, Value *Piv, bool Inverse) {
    call("laswp", {ref(NRHS), X, ref(N), integer(1), ref(N), Piv,
                   integer(Inverse ? -1 : 1)});
  }
  void gemm(Value *N, Value *NRHS, Value *X, Value *Y, Value *LDY, Value *Out) {
    call("gemm",
         {character('N'), character('T'), ref(N), ref(N), ref(NRHS), real(-1),
          X, ref(N), Y, ref(LDY), real(0), Out, ref(N)},
         2);
  }
};

std::string helperName(StringRef Kind, Function *Caller, bool Active) {
  return "__enzyme_getrs_" + Kind.str() + (Active ? "_factors_" : "_rhs_") +
         Caller->getName().str();
}

std::string cacheSuffix(GetrsCache Cache) {
  return std::string(Cache.Factors ? ".copyLU" : ".borrowLU") +
         (Cache.Pivots ? ".copyPivots" : ".borrowPivots") +
         (Cache.Solution ? ".copyX" : ".borrowX");
}
} // namespace

StructType *getGetrsTapeType(LLVMContext &C, const BlasInfo &Blas) {
  auto IT = Blas.intType(C);
  auto FP = Blas.fpType(C);
  // trans, n, nrhs, LU, lda, pivots, solution, ldx, valid, shadow ldb,
  // shadow lda. Cached matrices are contiguous; shadows retain the caller LD.
  return StructType::get(C, {Type::getInt8Ty(C), IT, IT, getUnqual(FP), IT,
                             getUnqual(IT), getUnqual(FP), IT,
                             Type::getInt1Ty(C), IT, IT});
}

Function *getOrInsertGetrsSnapshot(Module &M, Function *Caller,
                                   const BlasInfo &Blas, bool Active,
                                   GetrsCache Cache) {
  auto Tape = getGetrsTapeType(M.getContext(), Blas);
  std::string Name =
      helperName("snapshot", Caller, Active) + cacheSuffix(Cache);
  if (auto F = M.getFunction(Name))
    return F;
  auto F = Function::Create(FunctionType::get(Tape, {Tape}, false),
                            GlobalValue::InternalLinkage, Name, M);
  GetrsBuilder G(M, Caller, Blas, F);
  auto &B = G.B;
  auto In = F->getArg(0);
  auto N = B.CreateExtractValue(In, getrs::N);
  auto R = B.CreateExtractValue(In, getrs::NRHS);
  auto Empty = BasicBlock::Create(M.getContext(), "empty", F);
  auto Work = BasicBlock::Create(M.getContext(), "snapshot", F);
  auto Valid =
      B.CreateAnd(B.CreateExtractValue(In, getrs::Valid),
                  B.CreateAnd(B.CreateICmpSGT(N, ConstantInt::get(G.IT, 0)),
                              B.CreateICmpSGT(R, ConstantInt::get(G.IT, 0))));
  B.CreateCondBr(Valid, Work, Empty);
  B.SetInsertPoint(Empty);
  B.CreateRet(Constant::getNullValue(Tape));
  B.SetInsertPoint(Work);
  Value *Out = In;
  if (Cache.Factors) {
    auto A = G.allocate(G.FP, N, N, "getrs.lu");
    G.copy(A, B.CreateExtractValue(In, getrs::Factors), N, N,
           B.CreateExtractValue(In, getrs::LDA));
    Out = B.CreateInsertValue(Out, A, getrs::Factors);
    Out = B.CreateInsertValue(Out, N, getrs::LDA);
  }
  if (Cache.Pivots) {
    auto P = G.allocate(G.IT, N, ConstantInt::get(G.IT, 1), "getrs.pivots");
    G.loop(N, "pivot.copy", [&](Value *I) {
      B.CreateStore(
          B.CreateLoad(G.IT,
                       G.at(G.IT, B.CreateExtractValue(In, getrs::Pivots), I)),
          G.at(G.IT, P, I));
    });
    Out = B.CreateInsertValue(Out, P, getrs::Pivots);
  }
  if (Active && Cache.Solution) {
    auto X = G.allocate(G.FP, N, R, "getrs.solution");
    G.copy(X, B.CreateExtractValue(In, getrs::Solution), N, R,
           B.CreateExtractValue(In, getrs::LDX));
    Out = B.CreateInsertValue(Out, X, getrs::Solution);
    Out = B.CreateInsertValue(Out, N, getrs::LDX);
  }
  B.CreateRet(Out);
  return F;
}

Function *getOrInsertGetrsDerivative(Module &M, Function *Caller,
                                     const BlasInfo &Blas, bool Reverse,
                                     bool Active, bool FreeTape,
                                     GetrsCache Cache) {
  // Q is the forward row-swap sequence, so Q*A = L*U. Only the strictly
  // lower part of the factor tangent belongs to L; its unit diagonal is fixed.
  // The remaining packed entries belong to U. Forward mode forms dA*X (or
  // dA^T*X) through triangular products, subtracts it from dB, then solves.
  //
  // In reverse mode Y = op(A)^(-T)*bar(X). For a normal solve set W=Y,V=X;
  // for a transpose solve set W=X,V=Y. The packed factor adjoints are
  //   strict_lower(-(Q*W)*(U*V)^T) and upper(-(L^T*Q*W)*V^T).
  // They accumulate into the caller's shadow; the RHS shadow becomes Y.
  //
  // Forward mode receives borrowed primal pointers and never frees the tape.
  // Reverse mode owns only the copies selected by the overwrite analysis.
  // With multiple lanes only the final lane is instantiated with FreeTape=true.
  auto Tape = getGetrsTapeType(M.getContext(), Blas);
  std::string Name =
      helperName(Reverse ? "reverse" : "forward", Caller, Active) +
      cacheSuffix(Cache);
  if (FreeTape)
    Name += ".free";
  if (auto F = M.getFunction(Name))
    return F;
  auto FP = Blas.fpType(M.getContext());
  auto F = Function::Create(
      FunctionType::get(Type::getVoidTy(M.getContext()),
                        {Tape, getUnqual(FP), getUnqual(FP)}, false),
      GlobalValue::InternalLinkage, Name, M);
  GetrsBuilder G(M, Caller, Blas, F);
  auto &B = G.B;
  auto T = F->getArg(0);
  auto DA = F->getArg(1);
  auto DB = F->getArg(2);
  auto Trans = B.CreateExtractValue(T, getrs::Trans);
  auto N = B.CreateExtractValue(T, getrs::N);
  auto R = B.CreateExtractValue(T, getrs::NRHS);
  auto A = B.CreateExtractValue(T, getrs::Factors);
  auto LDA = B.CreateExtractValue(T, getrs::LDA);
  auto P = B.CreateExtractValue(T, getrs::Pivots);
  auto X = B.CreateExtractValue(T, getrs::Solution);
  auto LDX = B.CreateExtractValue(T, getrs::LDX);
  auto LDB = B.CreateExtractValue(T, getrs::ShadowLDB);
  auto LDDA = B.CreateExtractValue(T, getrs::ShadowLDA);
  auto End = BasicBlock::Create(M.getContext(), "end", F);
  auto Work = BasicBlock::Create(M.getContext(), "derivative", F);
  auto Valid =
      B.CreateAnd(B.CreateExtractValue(T, getrs::Valid),
                  B.CreateAnd(B.CreateICmpSGT(N, ConstantInt::get(G.IT, 0)),
                              B.CreateICmpSGT(R, ConstantInt::get(G.IT, 0))));
  B.CreateCondBr(Valid, Work, End);
  B.SetInsertPoint(Work);
  auto Normal = B.CreateOr(B.CreateICmpEQ(Trans, B.getInt8('N')),
                           B.CreateICmpEQ(Trans, B.getInt8('n')));
  if (Reverse) {
    auto Opposite = B.CreateSelect(Normal, B.getInt8('T'), B.getInt8('N'));
    G.solve(Opposite, N, R, A, LDA, P, DB, LDB);
  }
  if (Active) {
    auto W1 = G.allocate(FP, N, R, "getrs.work1");
    auto W2 = G.allocate(FP, N, R, "getrs.work2");
    if (Reverse) {
      auto W3 = G.allocate(FP, N, R, "getrs.work3");
      auto V = B.CreateSelect(Normal, X, DB);
      auto LDV = B.CreateSelect(Normal, LDX, LDB);
      auto W = B.CreateSelect(Normal, DB, X);
      auto LDW = B.CreateSelect(Normal, LDB, LDX);
      G.copy(W1, W, N, R, LDW);
      G.permute(W1, N, R, P, false);
      G.copy(W2, V, N, R, LDV);
      G.trmm('U', 'N', 'N', N, R, A, LDA, W2);
      G.copy(W3, W1, N, R, N);
      G.trmm('L', 'T', 'U', N, R, A, LDA, W3);
      auto Mat = G.allocate(FP, N, N, "getrs.factor.adjoint");
      for (unsigned Part = 0; Part < 2; ++Part) {
        G.gemm(N, R, Part ? W3 : W1, Part ? V : W2, Part ? LDV : N, Mat);
        G.loop(N, "adjoint.col", [&](Value *J) {
          G.loop(N, "adjoint.row", [&](Value *I) {
            auto InPart = Part ? B.CreateICmpULE(I, J) : B.CreateICmpUGT(I, J);
            auto Delta = B.CreateSelect(InPart, G.load(Mat, I, J, N),
                                        ConstantFP::get(FP, 0));
            G.store(B.CreateFAdd(G.load(DA, I, J, LDDA), Delta), DA, I, J,
                    LDDA);
          });
        });
      }
      CreateDealloc(B, Mat);
      CreateDealloc(B, W3);
    } else {
      auto DL = G.allocate(FP, N, N, "getrs.lower.tangent");
      G.loop(N, "lower.col", [&](Value *J) {
        G.loop(N, "lower.row", [&](Value *I) {
          auto V = B.CreateSelect(B.CreateICmpUGT(I, J), G.load(DA, I, J, LDDA),
                                  ConstantFP::get(FP, 0));
          G.store(V, DL, I, J, N);
        });
      });
      G.copy(W1, X, N, R, LDX);
      auto NB = BasicBlock::Create(M.getContext(), "normal", F);
      auto TB = BasicBlock::Create(M.getContext(), "transpose", F);
      auto Join = BasicBlock::Create(M.getContext(), "combine", F);
      B.CreateCondBr(Normal, NB, TB);
      B.SetInsertPoint(NB);
      G.copy(W2, X, N, R, LDX);
      G.trmm('U', 'N', 'N', N, R, A, LDA, W1);
      G.trmm('L', 'N', 'N', N, R, DL, N, W1);
      G.trmm('U', 'N', 'N', N, R, DA, LDDA, W2);
      G.trmm('L', 'N', 'U', N, R, A, LDA, W2);
      B.CreateBr(Join);
      B.SetInsertPoint(TB);
      G.permute(W1, N, R, P, false);
      G.copy(W2, W1, N, R, N);
      G.trmm('L', 'T', 'N', N, R, DL, N, W1);
      G.trmm('U', 'T', 'N', N, R, A, LDA, W1);
      G.trmm('L', 'T', 'U', N, R, A, LDA, W2);
      G.trmm('U', 'T', 'N', N, R, DA, LDDA, W2);
      B.CreateBr(Join);
      B.SetInsertPoint(Join);
      G.loop(R, "sum.col", [&](Value *J) {
        G.loop(N, "sum.row", [&](Value *I) {
          G.store(B.CreateFAdd(G.load(W1, I, J, N), G.load(W2, I, J, N)), W1, I,
                  J, N);
        });
      });
      auto Perm = BasicBlock::Create(M.getContext(), "unpermute", F);
      auto Sub = BasicBlock::Create(M.getContext(), "subtract", F);
      B.CreateCondBr(Normal, Perm, Sub);
      B.SetInsertPoint(Perm);
      G.permute(W1, N, R, P, true);
      B.CreateBr(Sub);
      B.SetInsertPoint(Sub);
      G.loop(R, "rhs.col", [&](Value *J) {
        G.loop(N, "rhs.row", [&](Value *I) {
          G.store(B.CreateFSub(G.load(DB, I, J, LDB), G.load(W1, I, J, N)), DB,
                  I, J, LDB);
        });
      });
      CreateDealloc(B, DL);
    }
    CreateDealloc(B, W1);
    CreateDealloc(B, W2);
  }
  if (!Reverse)
    G.solve(Trans, N, R, A, LDA, P, DB, LDB);
  if (Reverse && FreeTape) {
    if (Cache.Factors)
      CreateDealloc(B, A);
    if (Cache.Pivots)
      CreateDealloc(B, P);
    if (Active && Cache.Solution)
      CreateDealloc(B, X);
  }
  B.CreateBr(End);
  B.SetInsertPoint(End);
  B.CreateRetVoid();
  return F;
}
