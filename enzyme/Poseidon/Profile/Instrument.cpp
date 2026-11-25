//===- Instrument.cpp - profiling probes on a site clone ------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Instrument.h"

#include "Enzyme/Utils.h"
#include "IR/Utils.h"
#include "Pass/Session.h"
#include "Sites/Canonicalize.h"
#include "Sites/LaunchDescriptors.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"
#include "runtime/fpprofiler/FPProfilerCUDA.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Demangle/Demangle.h"
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
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <memory>
#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

Constant *getStringPointer(Module &M, StringRef gvName, StringRef contents) {
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = getInt8PtrTy(Ctx);
  GlobalVariable *gv = gvName.empty() ? nullptr : M.getNamedGlobal(gvName);
  if (!gv) {
    auto *strConst = ConstantDataArray::getString(Ctx, contents, true);
    gv = new GlobalVariable(
        M, strConst->getType(), true, GlobalValue::PrivateLinkage, strConst,
        gvName.empty() ? "poseidon.str" : gvName, nullptr,
        GlobalVariable::NotThreadLocal, Target::of(M).globalAddressSpace());
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    gv->setAlignment(Align(1));
  }
  Constant *ptr = gv;
  if (ptr->getType() != PtrTy)
    ptr = ConstantExpr::getAddrSpaceCast(ptr, PtrTy);
  return ptr;
}

// The profile record is keyed by this string, which both FP profiler runtimes
// use as the registry key and as the .fpprofile stem.
static Constant *getProfileNamePointer(Module &M, StringRef cloneName) {
  return getStringPointer(M, ("poseidon_site_" + cloneName).str(), cloneName);
}

FunctionCallee getProfilerHook(Module &M, StringRef name, FunctionType *FT) {
  FunctionCallee fc = M.getOrInsertFunction(name, FT);
  if (auto *fn = dyn_cast<Function>(fc.getCallee())) {
    fn->addFnAttr("enzyme_inactive");
    fn->addFnAttr(Attribute::NoFree);
  }
  return fc;
}

// The types of the profiler's hooks: (site name, slot, value, operand count,
// operands), (site name, slot, value, adjoint), and (site id, value) of the
// perturbation at `FpTy`.
static FunctionType *logValueType(LLVMContext &C) {
  Type *Ptr = getInt8PtrTy(C), *I64 = Type::getInt64Ty(C),
       *Double = Type::getDoubleTy(C);
  return FunctionType::get(Type::getVoidTy(C), {Ptr, I64, Double, I64, Ptr},
                           false);
}

static FunctionType *logGradientType(LLVMContext &C) {
  Type *Double = Type::getDoubleTy(C);
  return FunctionType::get(
      Type::getVoidTy(C),
      {getInt8PtrTy(C), Type::getInt64Ty(C), Double, Double}, false);
}

static FunctionType *perturbType(Type *FpTy) {
  return FunctionType::get(FpTy, {Type::getInt32Ty(FpTy->getContext()), FpTy},
                           false);
}

namespace {

struct ProbeBuilder {
  Module &M;
  LLVMContext &Ctx;
  const ProbeABI &P;
  unsigned siteId;
  StringRef cloneName;
  Constant *namePtr;

  Type *doubleTy() const { return Type::getDoubleTy(Ctx); }
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
      if (!P.PerturbF64.empty()) {
        FunctionCallee pert = getProfilerHook(
            M, isFloat ? P.PerturbF32 : P.PerturbF64, perturbType(FpTy));
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
      FunctionCallee logGrad =
          getProfilerHook(M, P.LogGradient, logGradientType(Ctx));
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
    // device heap.
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
  const ProbeABI &P = Target::of(M).probes();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *PtrTy = getInt8PtrTy(Ctx);
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

  if (!P.RuntimeAnchor.empty()) {
    // Referenced so that the build links the archive member that writes the
    // dumps out at exit.
    M.getOrInsertGlobal(P.RuntimeAnchor, Int32Ty);
  }

  ProbeBuilder PB{M,
                  Ctx,
                  P,
                  siteId,
                  clone.getName(),
                  getProfileNamePointer(M, clone.getName())};

  FunctionCallee logValue = getProfilerHook(M, P.LogValue, logValueType(Ctx));

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
        report_fatal_error("Unsupported operand type");
      Value *ptr = B.CreateGEP(operandArrayType, operandArray,
                               {ConstantInt::get(Int32Ty, 0),
                                ConstantInt::get(Int32Ty, operand.index())});
      B.CreateStore(operandValue, ptr);
    }
    Value *operandPtr = B.CreateGEP(
        operandArrayType, operandArray,
        {ConstantInt::get(Int32Ty, 0), ConstantInt::get(Int32Ty, 0)});
    // AMDGPU's stack is its own address space.
    if (operandPtr->getType() != PtrTy)
      operandPtr = B.CreateAddrSpaceCast(operandPtr, PtrTy);

    Value *nops = ConstantInt::get(SizeTy, numOperands);
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
  const ProbeABI &P = Target::of(M).probes();
  if (P.PublishStatic.empty()) {
    registerProfileStatic(M, clone.getName(), text);
    return;
  }
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = getInt8PtrTy(Ctx);

  Constant *namePtr = getProfileNamePointer(M, clone.getName());
  Constant *textPtr =
      getStringPointer(M, ("fpprofile_static_" + clone.getName()).str(), text);
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(Ctx), {PtrTy, PtrTy}, false);

  // Device: published by the site's own kernel through the pointer-keyed
  // registry, the only path from a device constant to the host writer.
  FunctionCallee probe = getProfilerHook(M, P.PublishStatic, FT);
  if (callsFunctionNamed(clone, P.PublishStatic))
    return;
  IRBuilder<> B(&*clone.getEntryBlock().getFirstInsertionPt());
  B.CreateCall(probe, {namePtr, textPtr})->setDoesNotThrow();
}

// Registered before main, so a site whose body never runs still carries its
// static data into the dump.
bool registerProfileStatic(Module &M, StringRef name, StringRef text) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *PtrTy = getInt8PtrTy(Ctx);

  Constant *namePtr = getProfileNamePointer(M, name);
  Constant *textPtr =
      getStringPointer(M, ("fpprofile_static_" + name).str(), text);
  FunctionType *FT = FunctionType::get(VoidTy, {PtrTy, PtrTy}, false);

  FunctionCallee reg =
      getProfilerHook(M, Target::of(M).probes().RegisterStatic, FT);
  std::string ctorName = ("poseidon_register_static_" + name).str();
  if (M.getFunction(ctorName))
    return false;
  Function *ctor = Function::Create(FunctionType::get(VoidTy, {}, false),
                                    GlobalValue::InternalLinkage, ctorName, &M);
  ctor->addFnAttr(Attribute::NoUnwind);
  IRBuilder<> B(BasicBlock::Create(Ctx, "entry", ctor));
  B.CreateCall(reg, {namePtr, textPtr})->setDoesNotThrow();
  B.CreateRetVoid();
  appendToGlobalCtors(M, ctor, 0);
  return true;
}

bool registerDeviceProfileTable(const Options &O, Module &M) {
  if (!O.Profiling.ProfileGenerate || Target::of(M).isDevice() ||
      M.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE))
    return false;
  std::optional<VariableRegistration> registration = variableRegistration(M);
  if (!registration)
    return false;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  Type *PtrTy = getInt8PtrTy(C);
  const uint64_t size = sizeof(PoseidonCUDAProfileTable);
  auto *shadowTy = ArrayType::get(Type::getInt8Ty(C), size);
  auto *shadow = new GlobalVariable(
      M, shadowTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
      Constant::getNullValue(shadowTy), POSEIDON_CUDA_PROFILE_TABLE);
  shadow->setAlignment(Align(8));
  registerDeviceVariable(M, *registration, shadow, POSEIDON_CUDA_PROFILE_TABLE,
                         size);

  Function *ctor = Function::Create(
      FunctionType::get(VoidTy, false), GlobalValue::InternalLinkage,
      "__poseidon_profile_register_module_table", &M);
  ctor->addFnAttr(Attribute::NoUnwind);
  IRBuilder<> B(BasicBlock::Create(C, "entry", ctor));
  auto [functions, slots] = readProfileTable(O, M);
  B.CreateCall(M.getOrInsertFunction(
                   "__poseidon_profile_register_module",
                   FunctionType::get(
                       VoidTy, {PtrTy, B.getInt32Ty(), B.getInt32Ty()}, false)),
               {shadow, B.getInt32(functions), B.getInt32(slots)});
  B.CreateRetVoid();
  // After the CUDA module constructor, so that a unit loaded once the profiler
  // runs is registered with the CUDA runtime before it is bound.
  appendToGlobalCtors(M, ctor, 65535);
  return true;
}

// Appends to `out` every shared global `V` refers to, through constant
// expressions.
static void collectShared(const Target &T, const Value *V,
                          SetVector<const GlobalVariable *> &out) {
  if (const auto *G = dyn_cast<GlobalVariable>(V)) {
    if (T.isShared(*G))
      out.insert(G);
  } else if (const auto *CE = dyn_cast<ConstantExpr>(V)) {
    for (const Value *Op : CE->operands())
      collectShared(T, Op, out);
  }
}

void requireSharedMemoryLimit(Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return;
  const DataLayout &DL = M.getDataLayout();
  // Enzyme names the shadow of a global in the global's enzyme_shadow
  // metadata.
  DenseMap<const GlobalVariable *, const GlobalVariable *> primalOf;
  for (const GlobalVariable &G : M.globals())
    if (MDNode *MD = G.getMetadata("enzyme_shadow"))
      if (auto *S =
              mdconst::dyn_extract_or_null<GlobalVariable>(MD->getOperand(0)))
        primalOf[S] = &G;
  const uint64_t limit = T.maxStaticSharedBytes();
  SmallVector<std::pair<const Function *, std::string>, 2> refused;
  for (const Function &K : M) {
    if (K.isDeclaration() || !T.isKernel(K))
      continue;
    SetVector<const Function *> funcs;
    SetVector<const GlobalVariable *> shared;
    bool unknownCall = false;
    funcs.insert(&K);
    for (size_t f = 0; f < funcs.size(); ++f) {
      for (const Instruction &I : instructions(*funcs[f])) {
        if (const auto *CB = dyn_cast<CallBase>(&I)) {
          const Function *callee = CB->getCalledFunction();
          if (callee && !callee->isDeclaration())
            funcs.insert(callee);
          unknownCall |= !callee && !CB->isInlineAsm();
        }
        for (const Value *Op : I.operands())
          collectShared(T, Op, shared);
      }
      // The target allocates a kernel that may make an unknown call the
      // shared memory of every function whose address is taken.
      if (f + 1 == funcs.size() && unknownCall &&
          T.allocatesAddressTakenShared())
        for (const Function &F : M)
          if (!F.isDeclaration() && F.hasAddressTaken())
            funcs.insert(&F);
    }
    uint64_t bytes = 0;
    for (const GlobalVariable *G : shared)
      bytes += DL.getTypeAllocSize(G->getValueType());
    if (bytes <= limit)
      continue;
    std::string own, shadows, others;
    raw_string_ostream ownOS(own), shadowsOS(shadows), othersOS(others);
    for (const GlobalVariable *G : shared) {
      const GlobalVariable *primal = primalOf.lookup(G);
      raw_string_ostream &os = !primal                   ? ownOS
                               : shared.contains(primal) ? shadowsOS
                                                         : othersOS;
      os << (os.str().empty() ? "" : ", ")
         << demangle((primal ? primal : G)->getName()) << " ("
         << DL.getTypeAllocSize(G->getValueType()) << " bytes)";
    }
    std::string msg;
    raw_string_ostream os(msg);
    os << displayName(K) << " reaches " << bytes
       << " bytes of static shared memory once profiled, above the " << limit
       << " a kernel can declare: its own buffers "
       << (own.empty() ? "none" : own) << "; the shadows of its buffers "
       << (shadows.empty() ? "none" : shadows)
       << "; the shadows of other kernels' buffers "
       << (others.empty() ? "none" : others)
       << ". Profile the kernel with smaller shared buffers, or the kernels "
          "whose shadows it reaches in another translation unit.";
    refused.emplace_back(&K, std::move(msg));
  }
  if (refused.empty())
    return;
  for (auto &[K, msg] : drop_end(refused))
    diagnose(*K, K->getSubprogram(), msg, DS_Error);
  reportUserError(*refused.back().first, refused.back().first->getSubprogram(),
                  refused.back().second);
}

void recordProfileTableSize(Session &S, Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return;
  const ProbeABI &P = T.probes();
  SmallPtrSet<const GlobalVariable *, 8> functions;
  uint64_t slots = 0;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      const Function *callee = CI ? CI->getCalledFunction() : nullptr;
      if (!callee || !is_contained({P.LogValue, P.LogGradient, P.BlockDims,
                                    P.PublishStatic},
                                   callee->getName()))
        continue;
      auto *name =
          dyn_cast<GlobalVariable>(CI->getArgOperand(0)->stripPointerCasts());
      if (!name)
        report_fatal_error("Poseidon: a device profiler probe names its "
                           "function through a value that is not a global");
      functions.insert(name);
      if (callee->getName() != P.LogValue && callee->getName() != P.LogGradient)
        continue;
      auto *idx = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      if (!idx)
        report_fatal_error("Poseidon: a device profiler probe has a slot "
                           "index that is not a constant");
      slots = std::max(slots, idx->getZExtValue() + 1);
    }
  if (!functions.empty())
    recordProfileTable(S, functions.size(), slots);
}

// The user's function a probe call serves. Profile generation calls the
// probes from functions it and Enzyme synthesize, all of local linkage; the
// nearest caller of other linkage is the kernel or device function the user
// wrote.
static const Function &userFunctionOf(const Function &F) {
  SmallVector<const Function *, 8> work{&F};
  SmallPtrSet<const Function *, 8> seen{&F};
  for (size_t i = 0; i < work.size(); ++i) {
    const Function *G = work[i];
    if (!G->hasLocalLinkage())
      return *G;
    for (const User *U : G->users())
      if (const auto *call = dyn_cast<CallBase>(U))
        if (seen.insert(call->getFunction()).second)
          work.push_back(call->getFunction());
  }
  return F;
}

// A probe compiled for a processor the device profiler cannot serve: an error
// at the user's function.
[[noreturn]] static void refuseProbe(const Target &target,
                                     const Function &caller) {
  const Function &F = userFunctionOf(caller);
  reportUserError(F, F.getSubprogram(), target.profilerMismatch(F));
}

bool linkDeviceProfiler(Module &M, bool profileGenerate) {
  const Target &target = Target::of(M);
  if (!profileGenerate || !target.isDevice())
    return false;
  if (GlobalVariable *T = M.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE))
    if (!T->isDeclaration())
      return false;
  // Without the target's device profiler the probes would stay external and
  // the program would fail only at its link.
  StringRef bitcode = target.profilerBitcode();
  if (bitcode.empty()) {
    const ProbeABI &P = target.probes();
    const StringRef probes[] = {
        P.LogValue,  P.LogGradient,   P.PerturbF64,          P.PerturbF32,
        P.BlockDims, P.PublishStatic, P.OverlappingArguments};
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        if (const auto *call = dyn_cast<CallBase>(&I))
          if (const Function *callee = call->getCalledFunction();
              callee && is_contained(probes, callee->getName()))
            refuseProbe(target, F);
    return false;
  }
  Expected<std::unique_ptr<Module>> parsed = parseBitcodeFile(
      MemoryBufferRef(bitcode, "FPProfilerDevice.bc"), M.getContext());
  if (!parsed)
    report_fatal_error(Twine("Poseidon: the device profiler bitcode does not "
                             "load: ") +
                       toString(parsed.takeError()));
  Module &L = **parsed;
  GlobalVariable *libTable = L.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE);
  if (!libTable ||
      L.getDataLayout().getTypeAllocSize(libTable->getValueType()) !=
          sizeof(PoseidonCUDAProfileTable))
    report_fatal_error("Poseidon: the device profiler bitcode does not define "
                       "the table the host runtime writes");
  // A library built for one processor serves only the probes compiled for it.
  for (const Function &F : L) {
    const Function *probe = F.isDeclaration() || F.hasLocalLinkage()
                                ? nullptr
                                : M.getFunction(F.getName());
    if (!probe || !probe->isDeclaration())
      continue;
    for (const User *U : probe->users())
      if (const auto *call = dyn_cast<CallBase>(U))
        if (!target.profilerMismatch(*call->getFunction()).empty())
          refuseProbe(target, *call->getFunction());
  }

  L.setTargetTriple(M.getTargetTriple());
  L.setDataLayout(M.getDataLayout());
  // The library's module flags, producer and keep-alive lists describe its own
  // compile; the module keeps its own, and takes only what its probes reach.
  if (NamedMDNode *moduleFlags = L.getModuleFlagsMetadata())
    L.eraseNamedMetadata(moduleFlags);
  if (NamedMDNode *ident = L.getNamedMetadata("llvm.ident"))
    L.eraseNamedMetadata(ident);
  for (StringRef used : {"llvm.used", "llvm.compiler.used"})
    if (GlobalVariable *G = L.getNamedGlobal(used))
      G->eraseFromParent();
  // Code generation follows the GPU the module is compiled for.
  for (Function &F : L) {
    F.removeFnAttr("target-cpu");
    F.removeFnAttr("target-features");
  }

  StringSet<> own;
  for (const GlobalValue &G : M.global_values())
    if (!G.isDeclaration())
      own.insert(G.getName());
  SmallVector<std::string, 16> linked;
  for (const GlobalValue &G : L.global_values())
    if (!G.isDeclaration() && !G.hasLocalLinkage() &&
        !own.contains(G.getName()))
      linked.push_back(G.getName().str());

  // The host half registers the table whether or not a probe reads it.
  GlobalVariable *table = M.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE);
  if (!table)
    table = new GlobalVariable(
        M, libTable->getValueType(), /*isConstant=*/false,
        GlobalValue::ExternalLinkage, nullptr, POSEIDON_CUDA_PROFILE_TABLE,
        nullptr, GlobalValue::NotThreadLocal, libTable->getAddressSpace());
  appendToCompilerUsed(M, {table});

  if (Linker::linkModules(M, std::move(*parsed), Linker::Flags::LinkOnlyNeeded))
    report_fatal_error("Poseidon: linking the device profiler failed");

  // Local to the module, so relocatable device code links one copy per unit;
  // the table is merged across units instead, because the linker wrapper
  // registers it by name in the linked image.
  for (const std::string &name : linked)
    if (GlobalValue *G = M.getNamedValue(name); G && !G->isDeclaration())
      G->setLinkage(name == POSEIDON_CUDA_PROFILE_TABLE
                        ? GlobalValue::WeakODRLinkage
                        : GlobalValue::InternalLinkage);
  return true;
}

// The generated logging code declares the probes on first use without
// enzyme_inactive, and with only a declaration in the TU Enzyme demands a
// derivative ("No augmented forward pass found"); so they are declared first,
// with the signatures the generated ones take. The runtime's own
// definitions cannot carry the attribute: __has_attribute(enzyme_inactive) is
// false unless the clang plugin is loaded, so under -fpass-plugin it is lost.
void predeclareInactiveProfilerProbes(Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return;
  const ProbeABI &P = T.probes();
  LLVMContext &Ctx = M.getContext();
  getProfilerHook(M, P.LogValue, logValueType(Ctx));
  getProfilerHook(M, P.LogGradient, logGradientType(Ctx));
  // The perturbation hooks are inactive too: nothing reads a derivative of the
  // perturbation, and a second run of the pass would otherwise differentiate
  // the calls the first run injected.
  getProfilerHook(M, P.PerturbF64, perturbType(Type::getDoubleTy(Ctx)));
  getProfilerHook(M, P.PerturbF32, perturbType(Type::getFloatTy(Ctx)));
}

// A static constructor cannot make the call: the runtime publishes its device
// tables with cudaMemcpyToSymbol, and a dynamic initializer can run before
// __cudaRegisterFatBinary has registered the device module in a multi-TU
// binary ("invalid device symbol"). main runs after every static constructor.
//
// Only the host half of an offload unit, kernels or not, links the device
// profiler. Erring towards injection is deliberate: a wrongly skipped
// injection loses the profile silently, a wrongly kept one fails at link.
bool injectHostProfilerInit(Module &M, bool print) {
  if (Target::of(M).isDevice())
    return false;
  // A host-only build links no CUDA profiler runtime, so the call injected
  // below would be an undefined reference to __poseidon_profile_init_cuda.
  OffloadRuntime R = offloadRuntimeOf(M);
  if (R == OffloadRuntime::None)
    return false;
  Function *mainF = M.getFunction("main");
  if (!mainF || mainF->isDeclaration())
    return false;
  LLVMContext &Ctx = M.getContext();
  FunctionCallee initFn =
      getProfilerHook(M, deviceProfilerInit(),
                      FunctionType::get(Type::getVoidTy(Ctx), {}, false));
  if (callsFunctionNamed(*mainF, deviceProfilerInit()))
    return false;
  IRBuilder<> B(&*mainF->getEntryBlock().getFirstInsertionPt());
  B.CreateCall(initFn, {})->setDoesNotThrow();
  if (print)
    llvm::errs()
        << "[poseidon] FP profiler CUDA registration injected at main\n";
  return true;
}

// The probe is keyed by the poseidon_site_<name> global the value probes use,
// which exists only after AD.
void injectBlockDimProbes(Module &M, bool print) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return;
  const ProbeABI &P = T.probes();
  Function *logFn = M.getFunction(P.LogValue);
  if (!logFn)
    return;
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = getInt8PtrTy(Ctx);
  FunctionCallee probeFn = getProfilerHook(
      M, P.BlockDims, FunctionType::get(Type::getVoidTy(Ctx), {PtrTy}, false));

  unsigned placed = 0;
  for (Function &F : M) {
    if (F.isDeclaration() || callsFunctionNamed(F, P.BlockDims))
      continue;
    Value *nameArg = nullptr;
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I);
          CI && getFunctionFromCall(CI) == logFn) {
        nameArg = CI->getArgOperand(0);
        break;
      }
    if (!nameArg)
      continue;
    // Rebuilt at the entry block from the global, as the log call's operand
    // may not dominate it; the runtime keys the record by that global.
    auto *gv = dyn_cast<GlobalVariable>(nameArg->stripPointerCasts());
    if (!gv)
      report_fatal_error("Poseidon: a value probe in " + F.getName() +
                         " has no poseidon_site_ global as its name operand");
    IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
    Value *nameAtEntry = B.CreateInBoundsGEP(gv->getValueType(), gv,
                                             {B.getInt32(0), B.getInt32(0)});
    if (nameAtEntry->getType() != PtrTy)
      nameAtEntry = B.CreateAddrSpaceCast(nameAtEntry, PtrTy);
    B.CreateCall(probeFn, {nameAtEntry})->setDoesNotThrow();
    ++placed;
  }
  if (print && placed)
    llvm::errs() << "[poseidon] block-geometry probe placed in " << placed
                 << " profiled function(s)\n";
}

} // namespace poseidon
