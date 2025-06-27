//===- Enzyme.cpp - Automatic Differentiation Transformation Pass  -------===//
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
// This file contains Enzyme, a transformation pass that takes replaces calls
// to function calls to *__enzyme_autodiff* with a call to the derivative of
// the function passed as the first argument.
//
//===----------------------------------------------------------------------===//
#define private public
#include "llvm/IR/Module.h"
#undef private
#include <llvm/Config/llvm-config.h>
#include <memory>

#include "llvm/ADT/StringRef.h"
#include <dlfcn.h>

#if LLVM_VERSION_MAJOR >= 16
#define private public
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#undef private
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include <optional>
#if LLVM_VERSION_MAJOR <= 16
#include "llvm/ADT/Optional.h"
#endif
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Transforms/IPO/GlobalOpt.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-reactant-intrinsic"

llvm::cl::opt<std::string>
    Passes("raising-plugin-path", cl::init(""), cl::Hidden,
           cl::desc("Print before and after fns for autodiff"));

namespace {

constexpr char cudaLaunchSymbolName[] = "cudaLaunchKernel";
constexpr char kernelPrefix[] = "__mlir_launch_kernel_";
constexpr char kernelCoercedPrefix[] = "__mlir_launch_coerced_kernel_";

constexpr char cudaPushConfigName[] = "__cudaPushCallConfiguration";
constexpr char cudaPopConfigName[] = "__cudaPopCallConfiguration";

SmallVector<CallBase *> gatherCallers(Function *F) {
  if (!F)
    return {};
  SmallVector<CallBase *> ToHandle;
  for (auto User : F->users())
    if (auto CI = dyn_cast<CallBase>(User))
      if (CI->getCalledFunction() == F)
        ToHandle.push_back(CI);
  return ToHandle;
}

void fixup(Module &M) {
  auto LaunchKernelFunc = M.getFunction(cudaLaunchSymbolName);
  if (!LaunchKernelFunc)
    return;

  SmallPtrSet<CallBase *, 8> CoercedKernels;
  for (CallBase *CI : gatherCallers(LaunchKernelFunc)) {
    IRBuilder<> Builder(CI);
    auto FuncPtr = CI->getArgOperand(0);
    auto GridDim1 = CI->getArgOperand(1);
    auto GridDim2 = CI->getArgOperand(2);
    auto BlockDim1 = CI->getArgOperand(3);
    auto BlockDim2 = CI->getArgOperand(4);
    auto ArgPtr = CI->getArgOperand(5);
    auto SharedMemSize = CI->getArgOperand(6);
    auto StreamPtr = CI->getArgOperand(7);
    SmallVector<Value *> Args = {
        FuncPtr,   GridDim1,      GridDim2,  BlockDim1,
        BlockDim2, SharedMemSize, StreamPtr,
    };
    auto StubFunc = cast<Function>(CI->getArgOperand(0));

    size_t idx = 0;
    for (auto &Arg : StubFunc->args()) {
      auto gep = Builder.CreateConstInBoundsGEP1_64(
          llvm::PointerType::getUnqual(CI->getContext()), ArgPtr, idx);
      auto ld = Builder.CreateLoad(
          llvm::PointerType::getUnqual(CI->getContext()), gep);
      ld = Builder.CreateLoad(Arg.getType(), ld);
      Args.push_back(ld);
      idx++;
    }
    SmallVector<Type *> ArgTypes;
    for (Value *V : Args)
      ArgTypes.push_back(V->getType());
    auto MlirLaunchFunc = Function::Create(
        FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes,
                          /*isVarAtg=*/false),
        llvm::GlobalValue::ExternalLinkage,
        kernelCoercedPrefix + StubFunc->getName(), M);

    CoercedKernels.insert(Builder.CreateCall(MlirLaunchFunc, Args));
    if (auto II = dyn_cast<InvokeInst>(CI)) {
      Builder.CreateBr(II->getNormalDest());
      II->getUnwindDest()->removePredecessor(II->getParent());
    }
    CI->eraseFromParent();
  }

  SmallVector<Function *> InlinedStubs;
  for (CallBase *CI : CoercedKernels) {
    Function *StubFunc = cast<Function>(CI->getArgOperand(0));
    for (User *callee : StubFunc->users()) {
      if (auto *CI = dyn_cast<CallBase>(callee)) {
        if (CI->getCalledFunction() == StubFunc) {
          InlineFunctionInfo IFI;
          InlineResult Res =
              InlineFunction(*CI, IFI, /*MergeAttributes=*/false);
          assert(Res.isSuccess());
          InlinedStubs.push_back(StubFunc);
          continue;
        }
      }
    }
  }
  for (Function *F : InlinedStubs) {
    F->erase(F->begin(), F->end());
    BasicBlock *BB = BasicBlock::Create(F->getContext(), "entry", F);
    ReturnInst::Create(F->getContext(), nullptr, BB->begin());
  }

  CoercedKernels.clear();
  DenseMap<Function *, SmallVector<AllocaInst *, 6>> FuncAllocas;
  auto PushConfigFunc = M.getFunction(cudaPushConfigName);
  for (CallBase *CI : gatherCallers(PushConfigFunc)) {
    Function *TheFunc = CI->getFunction();
    IRBuilder<> IRB(&TheFunc->getEntryBlock(),
                    TheFunc->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
    auto Allocas = FuncAllocas.lookup(TheFunc);
    if (Allocas.empty()) {
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "griddim64"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt32Ty(), nullptr, "griddim32"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "blockdim64"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt32Ty(), nullptr, "blockdim32"));
      Allocas.push_back(
          IRB.CreateAlloca(IRB.getInt64Ty(), nullptr, "shmem_size"));
      Allocas.push_back(IRB.CreateAlloca(IRB.getPtrTy(), nullptr, "stream"));
      FuncAllocas.insert_or_assign(TheFunc, Allocas);
      llvm::errs() << " CI: making allocas for  " << *CI << "\n";
    }
    IRB.SetInsertPoint(CI);
    if (CI->arg_size() != Allocas.size()) {
      llvm::errs() << " size mismatch on: " << *CI << "\n";
    }
    for (auto [Arg, Alloca] : llvm::zip_equal(CI->args(), Allocas))
      IRB.CreateStore(Arg, Alloca);
  }
  auto PopConfigFunc = M.getFunction(cudaPopConfigName);
  for (CallBase *PopCall : gatherCallers(PopConfigFunc)) {
    Function *TheFunc = PopCall->getFunction();
    auto Allocas = FuncAllocas.lookup(TheFunc);
    if (Allocas.empty()) {
      continue;
    }

    CallBase *KernelLaunch = PopCall;
    Instruction *It = PopCall;
    do {
      It = It->getNextNonDebugInstruction();
      KernelLaunch = dyn_cast<CallInst>(It);
    } while (!It->isTerminator() &&
             !(KernelLaunch && KernelLaunch->getCalledFunction() &&
               KernelLaunch->getCalledFunction()->getName().starts_with(
                   kernelCoercedPrefix)));

    assert(!It->isTerminator());

    IRBuilder<> IRB(PopCall);

    for (auto [Arg, Alloca] : llvm::zip(
             llvm::drop_begin(KernelLaunch->operand_values(), 1), Allocas)) {
      auto Load = cast<LoadInst>(Arg);
      LoadInst *NewLoad = IRB.CreateLoad(Arg->getType(), Alloca);
      Load->replaceAllUsesWith(NewLoad);
    }
    CoercedKernels.insert(KernelLaunch);
    // Replace with success
    PopCall->replaceAllUsesWith(IRB.getInt32(0));
    PopCall->eraseFromParent();
  }

  for (CallBase *PushCall : gatherCallers(PushConfigFunc)) {
    // Replace with success
    PushCall->replaceAllUsesWith(
        ConstantInt::get(IntegerType::get(PushCall->getContext(), 32), 0));
    PushCall->eraseFromParent();
  }
  for (CallBase *CI : CoercedKernels) {
    IRBuilder<> Builder(CI);
    auto FuncPtr = CI->getArgOperand(0);
    auto GridDim1 = CI->getArgOperand(1);
    auto GridDim2 = CI->getArgOperand(2);
    auto GridDimX = Builder.CreateTrunc(GridDim1, Builder.getInt32Ty());
    auto GridDimY = Builder.CreateLShr(
        GridDim1, ConstantInt::get(Builder.getInt64Ty(), 32));
    GridDimY = Builder.CreateTrunc(GridDimY, Builder.getInt32Ty());
    auto GridDimZ = GridDim2;
    auto BlockDim1 = CI->getArgOperand(3);
    auto BlockDim2 = CI->getArgOperand(4);
    auto BlockDimX = Builder.CreateTrunc(BlockDim1, Builder.getInt32Ty());
    auto BlockDimY = Builder.CreateLShr(
        BlockDim1, ConstantInt::get(Builder.getInt64Ty(), 32));
    BlockDimY = Builder.CreateTrunc(BlockDimY, Builder.getInt32Ty());
    auto BlockDimZ = BlockDim2;
    auto SharedMemSize = CI->getArgOperand(5);
    auto StreamPtr = CI->getArgOperand(6);
    SmallVector<Value *> Args = {
        FuncPtr,   GridDimX,  GridDimY,      GridDimZ,  BlockDimX,
        BlockDimY, BlockDimZ, SharedMemSize, StreamPtr,
    };
    auto StubFunc = cast<Function>(CI->getArgOperand(0));
    for (unsigned I = 7; I < CI->getNumOperands() - 1; I++)
      Args.push_back(CI->getArgOperand(I));
    SmallVector<Type *> ArgTypes;
    for (Value *V : Args)
      ArgTypes.push_back(V->getType());
    auto MlirLaunchFunc = Function::Create(
        FunctionType::get(Type::getVoidTy(M.getContext()), ArgTypes,
                          /*isVarAtg=*/false),
        llvm::GlobalValue::ExternalLinkage, kernelPrefix + StubFunc->getName(),
        M);

    Builder.CreateCall(MlirLaunchFunc, Args);
    CI->eraseFromParent();
  }
}

class ReactantBase {
public:
  std::vector<std::string> gpubins;
  ReactantBase(const std::vector<std::string> &gpubins) : gpubins(gpubins) {}

  bool run(Module &M) {
    bool changed = true;

    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << " pre fix: " << M << "\n";
    fixup(M);
    auto discard = M.getContext().shouldDiscardValueNames();
    M.getContext().setDiscardValueNames(false);
    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << " post fix: " << M << "\n";

    for (auto bin : gpubins) {
      SMDiagnostic Err;
      auto mod2 = llvm::parseIRFile(bin + ".re_export", Err, M.getContext());
      if (!mod2) {
        Err.print(/*ProgName=*/"LLVMToMLIR", llvm::errs());
        exit(1);
      }

      for (std::string T : {"", "f"}) {
        for (std::string name :
             {"sin",       "cos",     "tan",        "log2",    "exp",
              "exp2",      "exp10",   "cosh",       "sinh",    "tanh",
              "atan2",     "atan",    "asin",       "acos",    "log",
              "log10",     "log1p",   "acosh",      "asinh",   "atanh",
              "expm1",     "hypot",   "rhypot",     "norm3d",  "rnorm3d",
              "norm4d",    "rnorm4d", "norm",       "rnorm",   "cbrt",
              "rcbrt",     "j0",      "j1",         "y0",      "y1",
              "yn",        "jn",      "erf",        "erfinv",  "erfc",
              "erfcx",     "erfcinv", "normcdfinv", "normcdf", "lgamma",
              "ldexp",     "scalbn",  "frexp",      "modf",    "fmod",
              "remainder", "remquo",  "powi",       "tgamma",  "round",
              "fdim",      "ilogb",   "logb",       "isinf",   "pow",
              "sqrt",      "finite",  "fabs",       "fmax"}) {
          std::string nvname = "__nv_" + name;
          std::string llname = "llvm." + name + ".";
          std::string mathname = name;

          if (T == "f") {
            mathname += "f";
            nvname += "f";
            llname += "f32";
          } else {
            llname += "f64";
          }

          if (auto F = mod2->getFunction(llname)) {
            F->deleteBody();
          }
        }
      }
      {

        PassBuilder PB;
        LoopAnalysisManager LAM;
        FunctionAnalysisManager FAM;
        CGSCCAnalysisManager CGAM;
        ModuleAnalysisManager MAM;
        PB.registerModuleAnalyses(MAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        GlobalOptPass().run(*mod2, MAM);
      }
      for (auto &F : *mod2) {
        if (!F.empty())
          F.setLinkage(Function::LinkageTypes::InternalLinkage);
      }
      if (getenv("DEBUG_REACTANT"))
        llvm::errs() << " mod2: " << *mod2 << "\n";

      SmallVector<std::string> toInternalize;
      if (auto RF = M.getFunction("__cudaRegisterFunction")) {
        for (auto U : make_early_inc_range(RF->users())) {
          if (auto CI = dyn_cast<CallBase>(U)) {
            if (CI->getCalledFunction() != RF)
              continue;

            Value *F2 = CI->getArgOperand(1);
            Value *name = CI->getArgOperand(2);
            while (auto CE = dyn_cast<ConstantExpr>(F2)) {
              F2 = CE->getOperand(0);
            }
            while (auto CE = dyn_cast<ConstantExpr>(name)) {
              name = CE->getOperand(0);
            }
            StringRef nameVal;
            if (auto GV = dyn_cast<GlobalVariable>(name))
              if (GV->isConstant())
                if (auto C = GV->getInitializer())
                  if (auto CA = dyn_cast<ConstantDataArray>(C))
                    if (CA->getType()->getElementType()->isIntegerTy(8) &&
                        CA->isCString())
                      nameVal = CA->getAsCString();
            auto F22 = dyn_cast<Function>(F2);
            if (!F22)
              continue;

            if (nameVal.size())
              if (auto MF = mod2->getFunction(nameVal)) {
                MF->setName(F22->getName());
                F22->deleteBody();
                MF->setCallingConv(llvm::CallingConv::C);
                MF->setLinkage(Function::LinkageTypes::LinkOnceODRLinkage);
                toInternalize.push_back(MF->getName().str());
                CI->eraseFromParent();
              }
          }
        }
      }

      auto handler = M.getContext().getDiagnosticHandler();
      Linker L(M);
      L.linkInModule(std::move(mod2));
      M.getContext().setDiagnosticHandler(std::move(handler));
      for (auto name : toInternalize)
        if (auto F = M.getFunction(name)) {
          F->setLinkage(Function::LinkageTypes::InternalLinkage);
        }
    }

    if (getenv("DEBUG_REACTANT"))
      llvm::errs() << "post link: " << M << "\n";

    for (Function &F : make_early_inc_range(M)) {
      if (!F.empty())
        continue;
      if (F.getName() == "cudaMalloc") {
        continue;
        auto entry = BasicBlock::Create(F.getContext(), "entry", &F);
        IRBuilder<> B(entry);
      }
    }

    fixup(M);
    for (auto todel : {"__cuda_register_globals", "__cuda_module_ctor",
                       "__cuda_module_dtor"}) {
      if (auto F = M.getFunction(todel)) {
        F->replaceAllUsesWith(Constant::getNullValue(F->getType()));
        F->eraseFromParent();
      }
    }

    if (auto GV = M.getGlobalVariable("llvm.global_ctors")) {
      ConstantArray *CA = dyn_cast<ConstantArray>(GV->getInitializer());
      if (CA) {

        bool changed = false;
        SmallVector<Constant *> newOperands;
        for (Use &OP : CA->operands()) {
          if (isa<ConstantAggregateZero>(OP)) {
            changed = true;
            continue;
          }
          ConstantStruct *CS = cast<ConstantStruct>(OP);
          if (isa<ConstantPointerNull>(CS->getOperand(1))) {
            changed = true;
            continue;
          }
          newOperands.push_back(CS);
        }
        if (changed) {
          if (newOperands.size() == 0) {
            GV->eraseFromParent();
          } else {
            auto EltTy = newOperands[0]->getType();
            ArrayType *NewType = ArrayType::get(EltTy, newOperands.size());
            auto CT = ConstantArray::get(NewType, newOperands);

            // Create the new global variable.
            GlobalVariable *NG = new GlobalVariable(
                M, NewType, GV->isConstant(), GV->getLinkage(),
                /*init*/ CT, /*name*/ "", GV, GV->getThreadLocalMode(),
                GV->getAddressSpace());

            NG->copyAttributesFrom(GV);
            NG->takeName(GV);
            GV->replaceAllUsesWith(NG);
            GV->eraseFromParent();
          }
        }
      }
    }

    {
      PassBuilder PB;
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;
      PB.registerModuleAnalyses(MAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      GlobalOptPass().run(M, MAM);
    }

    auto lib = dlopen(Passes.c_str(), RTLD_LAZY | RTLD_DEEPBIND);
    if (!lib) {
      llvm::errs() << " could not open " << Passes.c_str() << " - " << dlerror()
                   << "\n";
    }
    auto sym = dlsym(lib, "runLLVMToMLIRRoundTrip");
    if (!sym) {
      llvm::errs() << " could not find sym\n";
    }
    auto runLLVMToMLIRRoundTrip = (std::string(*)(std::string))sym;
    if (runLLVMToMLIRRoundTrip) {
      std::string MStr;
      llvm::raw_string_ostream ss(MStr);
      ss << M;
      auto newMod = runLLVMToMLIRRoundTrip(MStr);
      M.dropAllReferences();

      M.getGlobalList().clear();
      M.getFunctionList().clear();
      M.getAliasList().clear();
      M.getIFuncList().clear();

      llvm::SMDiagnostic Err;
      auto llvmModule = llvm::parseIR(
          llvm::MemoryBufferRef(newMod, "conversion"), Err, M.getContext());

      if (!llvmModule) {
        llvm::errs() << " newMod: " << newMod << "\n";
        Err.print(/*ProgName=*/"LLVMToMLIR", llvm::errs());
        exit(1);
      }
      auto handler = M.getContext().getDiagnosticHandler();
      Linker L(M);
      L.linkInModule(std::move(llvmModule));
      M.getContext().setDiagnosticHandler(std::move(handler));
    }
    M.getContext().setDiscardValueNames(discard);

    return changed;
  }
};

} // namespace

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/Passes/PassPlugin.h"

class ReactantNewPM final : public ReactantBase,
                            public AnalysisInfoMixin<ReactantNewPM> {
  friend struct llvm::AnalysisInfoMixin<ReactantNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  ReactantNewPM(const std::vector<std::string> &gpubins)
      : ReactantBase(gpubins) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    return ReactantBase::run(M) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

class ExporterNewPM final : public AnalysisInfoMixin<ExporterNewPM> {
  friend struct llvm::AnalysisInfoMixin<ExporterNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  std::string firstfile;
  ExporterNewPM(std::string file) : firstfile(file) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    std::string filename = firstfile + ".re_export";

    std::error_code EC;
    llvm::raw_fd_ostream file(filename, EC); //, llvm::sys::fs::OF_Text);

    if (EC) {
      llvm::errs() << "Error opening file: " << EC.message() << "\n";
      exit(1);
    }

    file << M;
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

#undef DEBUG_TYPE
AnalysisKey ReactantNewPM::Key;
AnalysisKey ExporterNewPM::Key;

#include "llvm/Passes/PassBuilder.h"

extern "C" void registerExporter(llvm::PassBuilder &PB, std::string file) {
#if LLVM_VERSION_MAJOR >= 20
  auto loadPass =
      [=](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase)
#else
  auto loadPass = [=](ModulePassManager &MPM, OptimizationLevel Level)
#endif
  { MPM.addPass(ExporterNewPM(file)); };

  // TODO need for perf reasons to move Enzyme pass to the pre vectorization.
  PB.registerOptimizerEarlyEPCallback(loadPass);

  auto loadLTO = [loadPass](ModulePassManager &MPM, OptimizationLevel Level) {
#if LLVM_VERSION_MAJOR >= 20
    loadPass(MPM, Level, ThinOrFullLTOPhase::None);
#else
    loadPass(MPM, Level);
#endif
  };
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadLTO);
}

extern "C" void registerReactantAndPassPipeline(llvm::PassBuilder &PB,
                                                bool augment = false) {}

extern "C" void registerReactant(llvm::PassBuilder &PB,
                                 std::vector<std::string> gpubinaries) {

  llvm::errs() << " registering reactant\n";
#if LLVM_VERSION_MAJOR >= 20
  auto loadPass =
      [=](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase)
#else
  auto loadPass = [=](ModulePassManager &MPM, OptimizationLevel Level)
#endif
  { MPM.addPass(ReactantNewPM(gpubinaries)); };

  PB.registerPipelineParsingCallback(
      [=](llvm::StringRef Name, llvm::ModulePassManager &MPM,
          llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name == "reactant") {
          MPM.addPass(ReactantNewPM(gpubinaries));
          return true;
        }
        return false;
      });

  // TODO need for perf reasons to move Enzyme pass to the pre vectorization.
  PB.registerOptimizerEarlyEPCallback(loadPass);

  auto loadLTO = [loadPass](ModulePassManager &MPM, OptimizationLevel Level) {
#if LLVM_VERSION_MAJOR >= 20
    loadPass(MPM, Level, ThinOrFullLTOPhase::None);
#else
    loadPass(MPM, Level);
#endif
  };
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadLTO);
}

extern "C" void registerReactant2(llvm::PassBuilder &PB) {
  registerReactant(PB, {});
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ReactantNewPM", "v0.1", registerReactant2};
}
