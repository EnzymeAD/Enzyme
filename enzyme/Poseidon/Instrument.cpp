#include "Instrument.h"

#include "Optimize.h"
#include "ProfileRead.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ModRef.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace poseidon {

namespace {

Constant *getStringPointer(Module &M, StringRef gvName, StringRef contents,
                           bool isGPU) {
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = PointerType::getUnqual(Ctx);
  GlobalVariable *gv = M.getNamedGlobal(gvName);
  if (!gv) {
    auto *strConst = ConstantDataArray::getString(Ctx, contents, true);
    gv = new GlobalVariable(
        M, strConst->getType(), true, GlobalValue::PrivateLinkage, strConst,
        gvName, nullptr, GlobalVariable::NotThreadLocal, isGPU ? 1 : 0);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    gv->setAlignment(Align(1));
  }
  Constant *ptr = ConstantExpr::getInBoundsGetElementPtr(
      gv->getValueType(), gv,
      ArrayRef<Constant *>{ConstantInt::get(Type::getInt32Ty(Ctx), 0),
                           ConstantInt::get(Type::getInt32Ty(Ctx), 0)});
  if (ptr->getType() != PtrTy)
    ptr = ConstantExpr::getAddrSpaceCast(ptr, PtrTy);
  return ptr;
}

// The profile record is keyed by this string, which both FP profiler runtimes
// use as the registry key and as the .fpprofile stem.
Constant *getProfileNamePointer(Module &M, StringRef cloneName, bool isGPU) {
  return getStringPointer(M, ("poseidon_site_" + cloneName).str(), cloneName,
                          isGPU);
}

FunctionCallee getProfilerHook(Module &M, StringRef name, FunctionType *FT) {
  FunctionCallee fc = M.getOrInsertFunction(name, FT);
  if (auto *fn = dyn_cast<Function>(fc.getCallee())) {
    if (!fn->hasFnAttribute("enzyme_inactive"))
      fn->addFnAttr("enzyme_inactive");
    fn->addFnAttr(Attribute::NoFree);
  }
  return fc;
}

struct ProbeBuilder {
  Module &M;
  LLVMContext &Ctx;
  bool isGPU;
  unsigned siteId;
  StringRef cloneName;
  Constant *namePtr;

  Type *doubleTy() const { return Type::getDoubleTy(Ctx); }
  Type *ptrTy() const { return PointerType::getUnqual(Ctx); }
  Type *i64Ty() const { return Type::getInt64Ty(Ctx); }
  Type *i32Ty() const { return Type::getInt32Ty(Ctx); }

  // The identity probe for one profiled value. Its Enzyme custom derivative is
  // what carries the gradient record out of the reverse pass, and its augmented
  // forward is where the condition-number perturbation is applied, so that the
  // value flowing downstream and the value logged in the primal are both the
  // perturbed one while the adjoint chain stays untouched.
  Function *probeFor(Type *FpTy, size_t idx) {
    bool isFloat = FpTy->isFloatTy();
    std::string base =
        ("__poseidon_probe_" + cloneName + "_" + Twine(idx)).str();
    FunctionType *unaryFT = FunctionType::get(FpTy, {FpTy}, false);

    Function *aug = Function::Create(unaryFT, GlobalValue::InternalLinkage,
                                     base + "_aug", &M);
    aug->addFnAttr(Attribute::NoUnwind);
    {
      IRBuilder<> B(BasicBlock::Create(Ctx, "entry", aug));
      Value *v = aug->getArg(0);
      if (isGPU) {
        FunctionCallee pert = getProfilerHook(
            M,
            isFloat ? "poseidonProbePerturbCUDAf" : "poseidonProbePerturbCUDA",
            FunctionType::get(FpTy, {i32Ty(), FpTy}, false));
        v = B.CreateCall(pert, {ConstantInt::get(i32Ty(), siteId), v});
        cast<CallInst>(v)->setDoesNotThrow();
      }
      B.CreateRet(v);
    }

    FunctionType *revFT = FunctionType::get(FpTy, {FpTy, FpTy}, false);
    Function *rev = Function::Create(revFT, GlobalValue::InternalLinkage,
                                     base + "_rev", &M);
    rev->addFnAttr(Attribute::NoUnwind);
    {
      IRBuilder<> B(BasicBlock::Create(Ctx, "entry", rev));
      Value *v = B.CreateFPExt(rev->getArg(0), doubleTy());
      Value *d = B.CreateFPExt(rev->getArg(1), doubleTy());
      FunctionCallee logGrad = getProfilerHook(
          M, isGPU ? "poseidonLogGradCUDA" : "poseidonLogGrad",
          FunctionType::get(Type::getVoidTy(Ctx),
                            {ptrTy(), i64Ty(), doubleTy(), doubleTy()}, false));
      B.CreateCall(logGrad, {namePtr, ConstantInt::get(i64Ty(), idx), v, d})
          ->setDoesNotThrow();
      B.CreateRet(rev->getArg(1));
    }

    Function *probe =
        Function::Create(unaryFT, GlobalValue::InternalLinkage, base, &M);
    probe->addFnAttr(Attribute::NoInline);
    probe->addFnAttr(Attribute::NoUnwind);
    probe->addFnAttr(Attribute::WillReturn);
    // Not readnone: an identity the optimizer may fold away takes the profile
    // with it. Not argmem either, so the probe does not alias the kernel's own
    // memory traffic.
    probe->setMemoryEffects(MemoryEffects::inaccessibleMemOnly());
    // The reverse pass must be free to re-evaluate the probe rather than cache
    // its result. Without this every profiled value inside a loop becomes a
    // cache, and on a GPU those caches are device mallocs that exhaust the
    // 8 MiB device heap.
    probe->addFnAttr("enzyme_shouldrecompute");
    {
      IRBuilder<> B(BasicBlock::Create(Ctx, "entry", probe));
      B.CreateRet(probe->getArg(0));
    }
    probe->setMetadata("enzyme_augment",
                       MDTuple::get(Ctx, {ValueAsMetadata::get(aug)}));
    probe->setMetadata("enzyme_gradient",
                       MDTuple::get(Ctx, {ValueAsMetadata::get(rev)}));
    return probe;
  }
};

} // namespace

size_t instrumentForProfiling(Function &clone, unsigned siteId) {
  Module &M = *clone.getParent();
  LLVMContext &Ctx = M.getContext();
  bool isGPU = Triple(M.getTargetTriple()).isNVPTX();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *SizeTy = Type::getInt64Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);

  SmallVector<std::pair<Instruction *, size_t>, 32> targets;
  for (Instruction &I : instructions(clone)) {
    size_t idx;
    if (isOptimizable(I) && tryReadProfIdxMetadata(&I, idx))
      targets.emplace_back(&I, idx);
  }
  if (targets.empty())
    return 0;

  if (!isGPU) {
    // Reference the profiler runtime's registration variable so a host build
    // links the archive member that writes the dumps out at exit.
    M.getOrInsertGlobal("POSEIDON_PROFILE_RUNTIME_VAR", Int32Ty);
  }

  ProbeBuilder PB{M,
                  Ctx,
                  isGPU,
                  siteId,
                  clone.getName(),
                  getProfileNamePointer(M, clone.getName(), isGPU)};

  FunctionCallee logValue = getProfilerHook(
      M, isGPU ? "poseidonLogValueCUDA" : "poseidonLogValue",
      FunctionType::get(
          Type::getVoidTy(Ctx),
          {PtrTy, SizeTy, DoubleTy, isGPU ? SizeTy : Int32Ty, PtrTy}, false));

  IRBuilder<> allocaB(&*clone.getEntryBlock().getFirstInsertionPt());
  for (auto &[I, idx] : targets) {
    unsigned numOperands =
        isa<CallInst>(I) ? cast<CallInst>(I)->arg_size() : I->getNumOperands();
    ArrayType *operandArrayType = ArrayType::get(DoubleTy, numOperands);
    Value *operandArray = allocaB.CreateAlloca(operandArrayType);

    IRBuilder<> B(I->getNextNode());
    CallInst *probed = B.CreateCall(PB.probeFor(I->getType(), idx), {I});
    probed->setDebugLoc(I->getDebugLoc());
    I->replaceUsesWithIf(probed, [&](Use &U) { return U.getUser() != probed; });
    Value *loggedValue = B.CreateFPExt(probed, DoubleTy);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
    for (auto operand : enumerate(operands)) {
      Value *origOp = operand.value();
      Value *operandValue = nullptr;
      if (origOp->getType()->isFloatingPointTy())
        operandValue = B.CreateFPExt(origOp, DoubleTy);
      else if (origOp->getType()->isIntegerTy())
        operandValue = B.CreateSIToFP(origOp, DoubleTy);
      else
        llvm_unreachable("Unsupported operand type");
      Value *ptr = B.CreateGEP(operandArrayType, operandArray,
                               {ConstantInt::get(Int32Ty, 0),
                                ConstantInt::get(Int32Ty, operand.index())});
      B.CreateStore(operandValue, ptr);
    }
    Value *operandPtr = B.CreateGEP(
        operandArrayType, operandArray,
        {ConstantInt::get(Int32Ty, 0), ConstantInt::get(Int32Ty, 0)});

    Value *nops = isGPU ? ConstantInt::get(SizeTy, numOperands)
                        : ConstantInt::get(Int32Ty, numOperands);
    CallInst *logCall =
        B.CreateCall(logValue, {PB.namePtr, ConstantInt::get(SizeTy, idx),
                                loggedValue, nops, operandPtr});
    logCall->setDebugLoc(I->getDebugLoc());
  }

  return targets.size();
}

void emitProfileStaticData(Function &clone, StringRef text) {
  if (text.empty())
    return;
  Module &M = *clone.getParent();
  LLVMContext &Ctx = M.getContext();
  bool isGPU = Triple(M.getTargetTriple()).isNVPTX();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *PtrTy = PointerType::getUnqual(Ctx);

  Constant *namePtr = getProfileNamePointer(M, clone.getName(), isGPU);
  Constant *textPtr = getStringPointer(
      M, ("fpprofile_static_" + clone.getName()).str(), text, isGPU);
  FunctionType *FT = FunctionType::get(VoidTy, {PtrTy, PtrTy}, false);

  if (isGPU) {
    // Device: published by the site's own kernel through the pointer-keyed
    // registry, the only path from a device constant to the host writer.
    FunctionCallee probe = getProfilerHook(M, "poseidonProfileStaticCUDA", FT);
    for (Instruction &I : instructions(clone))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (auto *CF = CI->getCalledFunction())
          if (CF->getName() == "poseidonProfileStaticCUDA")
            return;
    IRBuilder<> B(&*clone.getEntryBlock().getFirstInsertionPt());
    B.CreateCall(probe, {namePtr, textPtr})->setDoesNotThrow();
    return;
  }

  // Host: registered before main, so a site whose body never runs still
  // carries its static data into the dump.
  FunctionCallee reg = getProfilerHook(M, "poseidonRegisterProfileStatic", FT);
  std::string ctorName = ("poseidon_register_static_" + clone.getName()).str();
  if (M.getFunction(ctorName))
    return;
  Function *ctor = Function::Create(FunctionType::get(VoidTy, {}, false),
                                    GlobalValue::InternalLinkage, ctorName, &M);
  ctor->addFnAttr(Attribute::NoUnwind);
  IRBuilder<> B(BasicBlock::Create(Ctx, "entry", ctor));
  B.CreateCall(reg, {namePtr, textPtr})->setDoesNotThrow();
  B.CreateRetVoid();
  appendToGlobalCtors(M, ctor, 0);
}

} // namespace poseidon
