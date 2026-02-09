//===- TypeAnalysis.cpp - Implementation of Type Analysis   ------------===//
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
// This file contains the implementation of Type Analysis, a utility for
// computing the underlying data type of LLVM values.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/ItaniumDemangle.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstIterator.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "llvm/IR/InlineAsm.h"

#include "../EnzymeLogic.h"
#include "../Utils.h"
#include "TypeAnalysis.h"

#include "../FunctionUtils.h"
#include "../LibraryFuncs.h"

#include "RustDebugInfo.h"
#include "TBAA.h"

#include <math.h>

#if LLVM_VERSION_MAJOR >= 14
#define getAttribute getAttributeAtIndex
#define hasAttribute hasAttributeAtIndex
#define addAttribute addAttributeAtIndex
#endif

using namespace llvm;

extern "C" {
/// Maximum offset for type trees to keep
llvm::cl::opt<int> MaxIntOffset("enzyme-max-int-offset", cl::init(100),
                                cl::Hidden,
                                cl::desc("Maximum type tree offset"));

llvm::cl::opt<unsigned> EnzymeMaxTypeDepth("enzyme-max-type-depth", cl::init(6),
                                           cl::Hidden,
                                           cl::desc("Maximum type tree depth"));

llvm::cl::opt<bool> EnzymePrintType("enzyme-print-type", cl::init(false),
                                    cl::Hidden,
                                    cl::desc("Print type analysis algorithm"));

llvm::cl::opt<bool> RustTypeRules("enzyme-rust-type", cl::init(false),
                                  cl::Hidden,
                                  cl::desc("Enable rust-specific type rules"));

llvm::cl::opt<bool> EnzymeStrictAliasing(
    "enzyme-strict-aliasing", cl::init(true), cl::Hidden,
    cl::desc("Assume strict aliasing of types / type stability"));
}

const llvm::StringMap<llvm::Intrinsic::ID> LIBM_FUNCTIONS = {
    {"sinc", Intrinsic::not_intrinsic},
    {"sincn", Intrinsic::not_intrinsic},
    {"cos", Intrinsic::cos},
    {"sin", Intrinsic::sin},
    {"tan", Intrinsic::not_intrinsic},
    {"acos", Intrinsic::not_intrinsic},
    {"__nv_frcp_rd", Intrinsic::not_intrinsic},
    {"__nv_frcp_rn", Intrinsic::not_intrinsic},
    {"__nv_frcp_ru", Intrinsic::not_intrinsic},
    {"__nv_frcp_rz", Intrinsic::not_intrinsic},
    {"__nv_drcp_rd", Intrinsic::not_intrinsic},
    {"__nv_drcp_rn", Intrinsic::not_intrinsic},
    {"__nv_drcp_ru", Intrinsic::not_intrinsic},
    {"__nv_drcp_rz", Intrinsic::not_intrinsic},
    {"asin", Intrinsic::not_intrinsic},
    {"__nv_asin", Intrinsic::not_intrinsic},
    {"atan", Intrinsic::not_intrinsic},
    {"atan2", Intrinsic::not_intrinsic},
    {"__nv_atan2", Intrinsic::not_intrinsic},
#if LLVM_VERSION_MAJOR >= 19
    {"cosh", Intrinsic::cosh},
    {"sinh", Intrinsic::sinh},
    {"tanh", Intrinsic::tanh},
#else
    {"cosh", Intrinsic::not_intrinsic},
    {"sinh", Intrinsic::not_intrinsic},
    {"tanh", Intrinsic::not_intrinsic},
#endif
    {"acosh", Intrinsic::not_intrinsic},
    {"asinh", Intrinsic::not_intrinsic},
    {"atanh", Intrinsic::not_intrinsic},
    {"exp", Intrinsic::exp},
    {"exp2", Intrinsic::exp2},
    {"exp10", Intrinsic::not_intrinsic},
    {"log", Intrinsic::log},
    {"log10", Intrinsic::log10},
    {"expm1", Intrinsic::not_intrinsic},
    {"log1p", Intrinsic::not_intrinsic},
    {"log2", Intrinsic::log2},
    {"logb", Intrinsic::not_intrinsic},
    {"pow", Intrinsic::pow},
    {"sqrt", Intrinsic::sqrt},
    {"cbrt", Intrinsic::not_intrinsic},
    {"hypot", Intrinsic::not_intrinsic},

    {"__mulsc3", Intrinsic::not_intrinsic},
    {"__muldc3", Intrinsic::not_intrinsic},
    {"__multc3", Intrinsic::not_intrinsic},
    {"__mulxc3", Intrinsic::not_intrinsic},

    {"__divsc3", Intrinsic::not_intrinsic},
    {"__divdc3", Intrinsic::not_intrinsic},
    {"__divtc3", Intrinsic::not_intrinsic},
    {"__divxc3", Intrinsic::not_intrinsic},

    {"Faddeeva_erf", Intrinsic::not_intrinsic},
    {"Faddeeva_erfc", Intrinsic::not_intrinsic},
    {"Faddeeva_erfcx", Intrinsic::not_intrinsic},
    {"Faddeeva_erfi", Intrinsic::not_intrinsic},
    {"Faddeeva_dawson", Intrinsic::not_intrinsic},
    {"Faddeeva_erf_re", Intrinsic::not_intrinsic},
    {"Faddeeva_erfc_re", Intrinsic::not_intrinsic},
    {"Faddeeva_erfcx_re", Intrinsic::not_intrinsic},
    {"Faddeeva_erfi_re", Intrinsic::not_intrinsic},
    {"Faddeeva_dawson_re", Intrinsic::not_intrinsic},
    {"erf", Intrinsic::not_intrinsic},
    {"erfi", Intrinsic::not_intrinsic},
    {"erfc", Intrinsic::not_intrinsic},
    {"erfinv", Intrinsic::not_intrinsic},

    {"__fd_sincos_1", Intrinsic::not_intrinsic},
    {"sincospi", Intrinsic::not_intrinsic},
    {"cmplx_inv", Intrinsic::not_intrinsic},

    // bessel functions
    {"j0", Intrinsic::not_intrinsic},
    {"j1", Intrinsic::not_intrinsic},
    {"jn", Intrinsic::not_intrinsic},
    {"y0", Intrinsic::not_intrinsic},
    {"y1", Intrinsic::not_intrinsic},
    {"yn", Intrinsic::not_intrinsic},
    {"tgamma", Intrinsic::not_intrinsic},
    {"lgamma", Intrinsic::not_intrinsic},
    {"logabsgamma", Intrinsic::not_intrinsic},
    {"ceil", Intrinsic::ceil},
    {"__nv_ceil", Intrinsic::ceil},
    {"floor", Intrinsic::floor},
    {"fmod", Intrinsic::not_intrinsic},
    {"trunc", Intrinsic::trunc},
    {"round", Intrinsic::round},
    {"rint", Intrinsic::rint},
    {"nearbyint", Intrinsic::nearbyint},
    {"remainder", Intrinsic::not_intrinsic},
    {"copysign", Intrinsic::copysign},
    {"nextafter", Intrinsic::not_intrinsic},
    {"nexttoward", Intrinsic::not_intrinsic},
    {"fdim", Intrinsic::not_intrinsic},
    {"fmax", Intrinsic::maxnum},
    {"fmin", Intrinsic::minnum},
    {"fabs", Intrinsic::fabs},
    {"fma", Intrinsic::fma},
    {"ilogb", Intrinsic::not_intrinsic},
    {"scalbn", Intrinsic::not_intrinsic},
    {"scalbln", Intrinsic::not_intrinsic},
    {"powi", Intrinsic::powi},
    {"cabs", Intrinsic::not_intrinsic},
    {"ldexp", Intrinsic::not_intrinsic},
    {"fmod", Intrinsic::not_intrinsic},
    {"finite", Intrinsic::not_intrinsic},
    {"isinf", Intrinsic::not_intrinsic},
    {"isnan", Intrinsic::not_intrinsic},
    {"lround", Intrinsic::lround},
    {"llround", Intrinsic::llround},
    {"lrint", Intrinsic::lrint},
    {"llrint", Intrinsic::llrint}};

static bool isItaniumEncoding(StringRef S) {
  // Itanium encoding requires 1 or 3 leading underscores, followed by 'Z'.
  return startsWith(S, "_Z") || startsWith(S, "___Z");
}

bool dontAnalyze(StringRef str) {
  if (isItaniumEncoding(str)) {
    if (str.empty())
      return false;

    ItaniumPartialDemangler Parser;
    char *data = (char *)malloc(str.size() + 1);
    memcpy(data, str.data(), str.size());
    data[str.size()] = 0;
    bool hasError = Parser.partialDemangle(data);
    if (hasError) {
      free(data);
      return false;
    }

    // auto basename = Parser.getFunctionBaseName(0, 0);
    // auto base = Parser.getFunctionDeclContextName(0, 0);
    // auto fn = Parser.getFunctionName(0, 0);
    // llvm::errs() << " err: " << base << " - " << basename << " fn - " << fn
    //              << "\n";
    free(data);
  }
  return false;
}

TypeAnalyzer::TypeAnalyzer(const FnTypeInfo &fn, TypeAnalysis &TA,
                           uint8_t direction)
    : MST(EnzymePrintType ? new ModuleSlotTracker(fn.Function->getParent())
                          : nullptr),
      notForAnalysis(getGuaranteedUnreachable(fn.Function)), intseen(),
      fntypeinfo(fn), interprocedural(TA), direction(direction), Invalid(false),
      PHIRecur(false),
      TLI(TA.Logic.PPC.FAM.getResult<TargetLibraryAnalysis>(*fn.Function)),
      DT(TA.Logic.PPC.FAM.getResult<DominatorTreeAnalysis>(*fn.Function)),
      PDT(TA.Logic.PPC.FAM.getResult<PostDominatorTreeAnalysis>(*fn.Function)),
      LI(TA.Logic.PPC.FAM.getResult<LoopAnalysis>(*fn.Function)),
      SE(TA.Logic.PPC.FAM.getResult<ScalarEvolutionAnalysis>(*fn.Function)) {

  assert(fntypeinfo.KnownValues.size() ==
         fntypeinfo.Function->getFunctionType()->getNumParams());

  // Add all instructions in the function
  for (BasicBlock &BB : *fntypeinfo.Function) {
    if (notForAnalysis.count(&BB))
      continue;
    for (Instruction &I : BB) {
      workList.insert(&I);
    }
  }
  // Add all operands referenced in the function
  // This is done to investigate any referenced globals/etc
  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (Instruction &I : BB) {
      for (auto &Op : I.operands()) {
        addToWorkList(Op);
      }
    }
  }
}

TypeAnalyzer::TypeAnalyzer(
    const FnTypeInfo &fn, TypeAnalysis &TA,
    const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &notForAnalysis,
    const TypeAnalyzer &Prev, uint8_t direction, bool PHIRecur)
    : MST(Prev.MST),
      notForAnalysis(notForAnalysis.begin(), notForAnalysis.end()), intseen(),
      fntypeinfo(fn), interprocedural(TA), direction(direction), Invalid(false),
      PHIRecur(PHIRecur), TLI(Prev.TLI), DT(Prev.DT), PDT(Prev.PDT),
      LI(Prev.LI), SE(Prev.SE) {
  assert(fntypeinfo.KnownValues.size() ==
         fntypeinfo.Function->getFunctionType()->getNumParams());
}

static SmallPtrSet<BasicBlock *, 1>
findLoopIndices(llvm::Value *val, LoopInfo &LI, DominatorTree &DT,
                SmallPtrSet<PHINode *, 1> &seen) {
  if (isa<Constant>(val))
    return {};
  if (auto CI = dyn_cast<CastInst>(val))
    return findLoopIndices(CI->getOperand(0), LI, DT, seen);
  if (auto CI = dyn_cast<UnaryOperator>(val))
    return findLoopIndices(CI->getOperand(0), LI, DT, seen);
  if (auto bo = dyn_cast<BinaryOperator>(val)) {
    auto inset0 = findLoopIndices(bo->getOperand(0), LI, DT, seen);
    auto inset1 = findLoopIndices(bo->getOperand(1), LI, DT, seen);
    inset0.insert(inset1.begin(), inset1.end());
    return inset0;
  }
  if (auto LDI = dyn_cast<LoadInst>(val)) {
    if (auto AI = dyn_cast<AllocaInst>(LDI->getPointerOperand())) {
      StoreInst *SI = nullptr;
      bool failed = false;
      for (auto u : AI->users()) {
        if (auto SIu = dyn_cast<StoreInst>(u)) {
          if (SI && SIu->getValueOperand() == AI) {
            failed = true;
            break;
          }
          SI = SIu;
        } else if (!isa<LoadInst>(u)) {
          if (!cast<Instruction>(u)->mayReadOrWriteMemory() &&
              cast<Instruction>(u)->use_empty())
            continue;
          if (auto CI = dyn_cast<CallBase>(u)) {
            if (auto F = CI->getCalledFunction()) {
              auto funcName = F->getName();
              if (funcName == "__kmpc_for_static_init_4" ||
                  funcName == "__kmpc_for_static_init_4u" ||
                  funcName == "__kmpc_for_static_init_8" ||
                  funcName == "__kmpc_for_static_init_8u") {
                continue;
              }
            }
          }
          failed = true;
          break;
        }
      }
      if (SI && !failed && DT.dominates(SI, LDI)) {
        return findLoopIndices(SI->getValueOperand(), LI, DT, seen);
      }
    }
  }
  if (auto pn = dyn_cast<PHINode>(val)) {
    auto L = LI.getLoopFor(pn->getParent());
    if (L && L->getHeader() == pn->getParent())
      return {pn->getParent()};
    if (seen.contains(pn))
      return {};
    SmallPtrSet<BasicBlock *, 1> ops;
    seen.insert(pn);
    for (unsigned i = 0; i < pn->getNumIncomingValues(); ++i) {
      auto a = pn->getIncomingValue(i);
      auto seti = findLoopIndices(a, LI, DT, seen);
      ops.insert(seti.begin(), seti.end());
    }
    return ops;
  }
  return {};
}

std::set<int64_t>
FnTypeInfo::knownIntegralValues(llvm::Value *val, const DominatorTree &DT,
                                std::map<Value *, std::set<int64_t>> &intseen,
                                ScalarEvolution &SE) const {
  if (auto constant = dyn_cast<ConstantInt>(val)) {
#if LLVM_VERSION_MAJOR > 14
    if (constant->getValue().getSignificantBits() > 64)
      return {};
#else
    if (constant->getValue().getMinSignedBits() > 64)
      return {};
#endif
    return {constant->getSExtValue()};
  }

  if (isa<ConstantPointerNull>(val)) {
    return {0};
  }

  assert(KnownValues.size() == Function->getFunctionType()->getNumParams());

  if (auto arg = dyn_cast<llvm::Argument>(val)) {
    auto found = KnownValues.find(arg);
    if (found == KnownValues.end()) {
      for (const auto &pair : KnownValues) {
        llvm::errs() << " KnownValues[" << *pair.first << "] - "
                     << pair.first->getParent()->getName() << "\n";
      }
      llvm::errs() << " arg: " << *arg << " - " << arg->getParent()->getName()
                   << "\n";
    }
    assert(found != KnownValues.end());
    return found->second;
  }

  if (intseen.find(val) != intseen.end())
    return intseen[val];
  intseen[val] = {};

  if (auto ci = dyn_cast<CastInst>(val)) {
    intseen[val] = knownIntegralValues(ci->getOperand(0), DT, intseen, SE);
  }

  auto insert = [&](int64_t v) {
    if (intseen[val].size() == 0) {
      intseen[val].insert(v);
    } else {
      if (intseen[val].size() == 1) {
        if (abs(*intseen[val].begin()) > MaxIntOffset) {
          if (abs(*intseen[val].begin()) > abs(v)) {
            intseen[val].clear();
            intseen[val].insert(v);
          } else {
            return;
          }
        } else {
          if (abs(v) > MaxIntOffset) {
            return;
          } else {
            intseen[val].insert(v);
          }
        }
      } else {
        if (abs(v) > MaxIntOffset) {
          return;
        } else {
          intseen[val].insert(v);
        }
      }
    }
  };
  if (auto II = dyn_cast<IntrinsicInst>(val)) {
    switch (II->getIntrinsicID()) {
#if LLVM_VERSION_MAJOR >= 12
    case Intrinsic::abs:
      for (auto val :
           knownIntegralValues(II->getArgOperand(0), DT, intseen, SE))
        insert(abs(val));
      break;
#endif
    case Intrinsic::nvvm_read_ptx_sreg_tid_x:
    case Intrinsic::nvvm_read_ptx_sreg_tid_y:
    case Intrinsic::nvvm_read_ptx_sreg_tid_z:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
    case Intrinsic::amdgcn_workitem_id_x:
    case Intrinsic::amdgcn_workitem_id_y:
    case Intrinsic::amdgcn_workitem_id_z:
      insert(0);
      break;
    default:
      break;
    }
  }
  if (auto LI = dyn_cast<LoadInst>(val)) {
    if (auto AI = dyn_cast<AllocaInst>(LI->getPointerOperand())) {
      StoreInst *SI = nullptr;
      bool failed = false;
      for (auto u : AI->users()) {
        if (auto SIu = dyn_cast<StoreInst>(u)) {
          if (SI && SIu->getValueOperand() == AI) {
            failed = true;
            break;
          }
          SI = SIu;
        } else if (!isa<LoadInst>(u)) {
          if (!cast<Instruction>(u)->mayReadOrWriteMemory() &&
              cast<Instruction>(u)->use_empty())
            continue;
          if (auto CI = dyn_cast<CallBase>(u)) {
            if (auto F = CI->getCalledFunction()) {
              auto funcName = F->getName();
              if (funcName == "__kmpc_for_static_init_4" ||
                  funcName == "__kmpc_for_static_init_4u" ||
                  funcName == "__kmpc_for_static_init_8" ||
                  funcName == "__kmpc_for_static_init_8u") {
                continue;
              }
            }
          }
          failed = true;
          break;
        }
      }
      if (SI && !failed && DT.dominates(SI, LI)) {
        for (auto val :
             knownIntegralValues(SI->getValueOperand(), DT, intseen, SE)) {
          insert(val);
        }
      }
    }
  }
  if (auto pn = dyn_cast<PHINode>(val)) {
    if (SE.isSCEVable(pn->getType()))
      if (auto S = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(pn))) {
        if (auto StartC = dyn_cast<SCEVConstant>(S->getStart())) {
          auto L = S->getLoop();
          auto BE = SE.getBackedgeTakenCount(L);
          if (BE != SE.getCouldNotCompute()) {
            if (auto Iters = dyn_cast<SCEVConstant>(BE)) {
              uint64_t ival = Iters->getAPInt().getZExtValue();
              // If strict aliasing and the loop header does not dominate all
              // blocks at low optimization levels the last "iteration" will
              // actually exit leading to one extra backedge that would be wise
              // to ignore.
              if (EnzymeStrictAliasing) {
                bool rotated = false;
                BasicBlock *Latch = L->getLoopLatch();
                rotated = Latch && L->isLoopExiting(Latch);
                if (!rotated) {
                  if (ival > 0)
                    ival--;
                }
              }

              uint64_t istart = 0;

              if (S->isAffine()) {
                if (auto StepC = dyn_cast<SCEVConstant>(S->getOperand(1))) {
                  APInt StartI = StartC->getAPInt();
                  APInt A = StepC->getAPInt();

                  if (A.sle(-1)) {
                    A = -A;
                    StartI = -StartI;
                  }

                  if (A.sge(1)) {
                    if (StartI.sge(MaxIntOffset)) {
                      ival = std::min(ival, (uint64_t)0);
                    } else {
                      ival = std::min(
                          ival,
                          (MaxIntOffset - StartI + A).udiv(A).getZExtValue());
                    }

                    if (StartI.slt(-MaxIntOffset)) {
                      istart = std::max(
                          istart,
                          (-MaxIntOffset - StartI).udiv(A).getZExtValue());
                    }

                  } else {
                    ival = std::min(ival, (uint64_t)0);
                  }
                } else {
                  ival = std::min(ival, (uint64_t)0);
                }
              }

              for (uint64_t i = istart; i <= ival; i++) {
                if (auto Val = dyn_cast<SCEVConstant>(S->evaluateAtIteration(
                        SE.getConstant(Iters->getType(), i, /*signed*/ false),
                        SE))) {
                  insert(Val->getAPInt().getSExtValue());
                }
              }
              return intseen[val];
            }
          }
        }
      }

    for (unsigned i = 0; i < pn->getNumIncomingValues(); ++i) {
      auto a = pn->getIncomingValue(i);
      auto b = pn->getIncomingBlock(i);

      // do not consider loop incoming edges
      if (pn->getParent() == b || DT.dominates(pn, b)) {
        continue;
      }

      auto inset = knownIntegralValues(a, DT, intseen, SE);

      // TODO this here is not fully justified yet
      for (auto pval : inset) {
        if (pval < 20 && pval > -20) {
          insert(pval);
        }
      }

      // if we are an iteration variable, suppose that it could be zero in that
      // range
      // TODO: could actually check the range intercepts 0
      if (auto bo = dyn_cast<BinaryOperator>(a)) {
        if (bo->getOperand(0) == pn || bo->getOperand(1) == pn) {
          if (bo->getOpcode() == BinaryOperator::Add ||
              bo->getOpcode() == BinaryOperator::Sub) {
            insert(0);
          }
        }
      }
    }
    return intseen[val];
  }

  if (auto bo = dyn_cast<BinaryOperator>(val)) {
    auto inset0 = knownIntegralValues(bo->getOperand(0), DT, intseen, SE);
    auto inset1 = knownIntegralValues(bo->getOperand(1), DT, intseen, SE);
    if (bo->getOpcode() == BinaryOperator::Mul) {

      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {

            insert(val0 * val1);
          }
        }
      }
      if (inset0.count(0) || inset1.count(0)) {
        intseen[val].insert(0);
      }
    }

    if (bo->getOpcode() == BinaryOperator::Add) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 + val1);
          }
        }
      }
    }
    if (bo->getOpcode() == BinaryOperator::Sub) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 - val1);
          }
        }
      }
    }

    if (bo->getOpcode() == BinaryOperator::SDiv) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 / val1);
          }
        }
      }
    }

    if (bo->getOpcode() == BinaryOperator::Shl) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 << val1);
          }
        }
      }
    }

    // TODO note C++ doesnt guarantee behavior of >> being arithmetic or logical
    //     and should replace with llvm apint internal
    if (bo->getOpcode() == BinaryOperator::AShr ||
        bo->getOpcode() == BinaryOperator::LShr) {
      if (inset0.size() == 1 || inset1.size() == 1) {
        for (auto val0 : inset0) {
          for (auto val1 : inset1) {
            insert(val0 >> val1);
          }
        }
      }
    }
  }

  return intseen[val];
}

/// Given a constant value, deduce any type information applicable
void getConstantAnalysis(Constant *Val, TypeAnalyzer &TA,
                         std::map<llvm::Value *, TypeTree> &analysis) {
  auto found = analysis.find(Val);
  if (found != analysis.end())
    return;

  auto &DL = TA.fntypeinfo.Function->getParent()->getDataLayout();

  // Undefined value is an anything everywhere
  if (isa<UndefValue>(Val) || isa<ConstantAggregateZero>(Val)) {
    analysis[Val].insert({-1}, BaseType::Anything);
    return;
  }

  // Null pointer is a pointer to anything, everywhere
  if (isa<ConstantPointerNull>(Val)) {
    TypeTree &Result = analysis[Val];
    Result.insert({-1}, BaseType::Pointer);
    Result.insert({-1, -1}, BaseType::Anything);
    return;
  }

  // Known pointers are pointers at offset 0
  if (isa<Function>(Val) || isa<BlockAddress>(Val)) {
    analysis[Val].insert({-1}, BaseType::Pointer);
    return;
  }

  // Any constants == 0 are considered Anything
  // other floats are assumed to be that type
  if (auto FP = dyn_cast<ConstantFP>(Val)) {
    if (FP->isExactlyValue(0.0)) {
      analysis[Val].insert({-1}, BaseType::Anything);
      return;
    }
    analysis[Val].insert({-1}, ConcreteType(FP->getType()->getScalarType()));
    return;
  }

  if (auto ci = dyn_cast<ConstantInt>(Val)) {
    // Constants in range [1, 4096] are assumed to be integral since
    // any float or pointers they may represent are ill-formed
    if (!ci->isNegative() && ci->getLimitedValue() >= 1 &&
        ci->getLimitedValue() <= 4096) {
      analysis[Val].insert({-1}, BaseType::Integer);
      return;
    }

    // Constants explicitly marked as negative that aren't -1 are considered
    // integral if >= -4096
    if (ci->isNegative() && !ci->isMinusOne() && ci->getValue().sge(-4096)) {
      analysis[Val].insert({-1}, BaseType::Integer);
      return;
    }

    // Values of size < 16 (half size) are considered integral
    // since they cannot possibly represent a float or pointer
    if (cast<IntegerType>(ci->getType())->getBitWidth() < 16) {
      analysis[Val].insert({-1}, BaseType::Integer);
      return;
    }
    // All other constant-ints could be any type
    analysis[Val].insert({-1}, BaseType::Anything);
    return;
  }

  // Type of an aggregate is the aggregation of
  // the subtypes
  if (auto CA = dyn_cast<ConstantAggregate>(Val)) {
    TypeTree &Result = analysis[Val];
    for (unsigned i = 0, size = CA->getNumOperands(); i < size; ++i) {
      assert(TA.fntypeinfo.Function);
      auto Op = CA->getOperand(i);
      // TODO check this for i1 constant aggregates packing/etc
      auto ObjSize = (TA.fntypeinfo.Function->getParent()
                          ->getDataLayout()
                          .getTypeSizeInBits(Op->getType()) +
                      7) /
                     8;

      Value *vec[2] = {
          ConstantInt::get(Type::getInt64Ty(Val->getContext()), 0),
          ConstantInt::get(Type::getInt32Ty(Val->getContext()), i),
      };
      auto g2 = GetElementPtrInst::Create(
          Val->getType(), UndefValue::get(getUnqual(Val->getType())), vec);
      APInt ai(DL.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
      g2->accumulateConstantOffset(DL, ai);
      // Using destructor rather than eraseFromParent
      //   as g2 has no parent
      delete g2;

      int Off = (int)ai.getLimitedValue();
      if (auto VT = dyn_cast<VectorType>(Val->getType()))
        if (VT->getElementType()->isIntegerTy(1))
          Off = i / 8;

      getConstantAnalysis(Op, TA, analysis);
      auto mid = analysis[Op];
      if (TA.fntypeinfo.Function->getParent()
              ->getDataLayout()
              .getTypeSizeInBits(CA->getType()) >= 16) {
        mid.ReplaceIntWithAnything();
      }

      Result |= mid.ShiftIndices(DL, /*init offset*/ 0,
                                 /*maxSize*/ ObjSize,
                                 /*addOffset*/ Off);
    }
    Result.CanonicalizeInPlace(
        (TA.fntypeinfo.Function->getParent()->getDataLayout().getTypeSizeInBits(
             CA->getType()) +
         7) /
            8,
        DL);
    return;
  }

  // Type of an sequence is the aggregation of
  // the subtypes
  if (auto CD = dyn_cast<ConstantDataSequential>(Val)) {
    TypeTree &Result = analysis[Val];
    for (unsigned i = 0, size = CD->getNumElements(); i < size; ++i) {
      assert(TA.fntypeinfo.Function);
      auto Op = CD->getElementAsConstant(i);
      // TODO check this for i1 constant aggregates packing/etc
      auto ObjSize = (TA.fntypeinfo.Function->getParent()
                          ->getDataLayout()
                          .getTypeSizeInBits(Op->getType()) +
                      7) /
                     8;

      Value *vec[2] = {
          ConstantInt::get(Type::getInt64Ty(Val->getContext()), 0),
          ConstantInt::get(Type::getInt32Ty(Val->getContext()), i),
      };
      auto g2 = GetElementPtrInst::Create(
          Val->getType(), UndefValue::get(getUnqual(Val->getType())), vec);
      APInt ai(DL.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
      g2->accumulateConstantOffset(DL, ai);
      // Using destructor rather than eraseFromParent
      //   as g2 has no parent
      delete g2;

      int Off = (int)ai.getLimitedValue();

      getConstantAnalysis(Op, TA, analysis);
      auto mid = analysis[Op];
      if (TA.fntypeinfo.Function->getParent()
              ->getDataLayout()
              .getTypeSizeInBits(CD->getType()) >= 16) {
        mid.ReplaceIntWithAnything();
      }
      Result |= mid.ShiftIndices(DL, /*init offset*/ 0,
                                 /*maxSize*/ ObjSize,
                                 /*addOffset*/ Off);

      Result |= mid;
    }
    Result.CanonicalizeInPlace(
        (TA.fntypeinfo.Function->getParent()->getDataLayout().getTypeSizeInBits(
             CD->getType()) +
         7) /
            8,
        DL);
    return;
  }

  // ConstantExprs are handled by considering the
  // equivalent instruction
  if (auto CE = dyn_cast<ConstantExpr>(Val)) {
    if (CE->isCast()) {
      if (CE->getType()->isPointerTy() && isa<ConstantInt>(CE->getOperand(0))) {
        analysis[Val] = TypeTree(BaseType::Anything).Only(-1, nullptr);
        return;
      }
      getConstantAnalysis(CE->getOperand(0), TA, analysis);
      analysis[Val] = analysis[CE->getOperand(0)];
      return;
    }
    if (CE->getOpcode() == Instruction::GetElementPtr) {
      TA.visitGEPOperator(*cast<GEPOperator>(CE));
      return;
    }

    auto I = CE->getAsInstruction();
    I->insertBefore(TA.fntypeinfo.Function->getEntryBlock().getTerminator());

    // Just analyze this new "instruction" and none of the others
    {
      TypeAnalyzer tmpAnalysis(TA.fntypeinfo, TA.interprocedural,
                               TA.notForAnalysis, TA);
      tmpAnalysis.visit(*I);
      analysis[Val] = tmpAnalysis.getAnalysis(I);

      if (tmpAnalysis.workList.remove(I)) {
        TA.workList.insert(CE);
      }
    }

    I->eraseFromParent();
    return;
  }

  if (auto GV = dyn_cast<GlobalVariable>(Val)) {

    if (GV->getName() == "__cxa_thread_atexit_impl") {
      analysis[Val] = TypeTree(BaseType::Pointer).Only(-1, nullptr);
      return;
    }

    // from julia code
    if (GV->getName() == "small_typeof" || GV->getName() == "jl_small_typeof") {
      TypeTree T;
      T.insert({-1}, BaseType::Pointer);
      T.insert({-1, -1}, BaseType::Pointer);
      analysis[Val] = T;
      return;
    }

    TypeTree &Result = analysis[Val];
    Result.insert({-1}, ConcreteType(BaseType::Pointer));

    // A fixed constant global is a pointer to its initializer
    if (GV->isConstant() && GV->hasInitializer()) {
      getConstantAnalysis(GV->getInitializer(), TA, analysis);
      Result |= analysis[GV->getInitializer()].Only(-1, nullptr);
      return;
    }
    if (!isa<StructType>(GV->getValueType()) ||
        !cast<StructType>(GV->getValueType())->isOpaque()) {
      auto globalSize = (DL.getTypeSizeInBits(GV->getValueType()) + 7) / 8;
      // Since halfs are 16bit (2 byte) and pointers are >=32bit (4 byte) any
      // Single byte object must be integral
      if (globalSize == 1) {
        Result.insert({-1, -1}, ConcreteType(BaseType::Integer));
        return;
      }
    }

    // Otherwise, we simply know that this is a pointer, and
    // not what it is a pointer to
    return;
  }

  // No other information can be ascertained
  analysis[Val] = TypeTree();
  return;
}

TypeTree TypeAnalyzer::getAnalysis(Value *Val) {
  // Integers with fewer than 16 bits (size of half)
  // must be integral, since it cannot possibly represent a float or pointer
  if (!isa<UndefValue>(Val) && Val->getType()->isIntegerTy() &&
      cast<IntegerType>(Val->getType())->getBitWidth() < 16)
    return TypeTree(BaseType::Integer).Only(-1, nullptr);
  if (auto C = dyn_cast<Constant>(Val)) {
    getConstantAnalysis(C, *this, analysis);
    return analysis[Val];
  }

  // Check that this value is from the function being analyzed
  if (auto I = dyn_cast<Instruction>(Val)) {
    if (I->getParent()->getParent() != fntypeinfo.Function) {
      llvm::errs() << " function: " << *fntypeinfo.Function << "\n";
      llvm::errs() << " instParent: " << *I->getParent()->getParent() << "\n";
      llvm::errs() << " inst: " << *I << "\n";
    }
    assert(I->getParent()->getParent() == fntypeinfo.Function);
  }
  if (auto Arg = dyn_cast<Argument>(Val)) {
    if (Arg->getParent() != fntypeinfo.Function) {
      llvm::errs() << " function: " << *fntypeinfo.Function << "\n";
      llvm::errs() << " argParent: " << *Arg->getParent() << "\n";
      llvm::errs() << " arg: " << *Arg << "\n";
    }
    assert(Arg->getParent() == fntypeinfo.Function);
  }

  // Return current results
  if (isa<Argument>(Val) || isa<Instruction>(Val))
    return analysis[Val];

  // Unhandled/unknown Value
  llvm::errs() << "Error Unknown Value: " << *Val << "\n";
  assert(0 && "Error Unknown Value: ");
  llvm_unreachable("Error Unknown Value: ");
  // return TypeTree();
}

void TypeAnalyzer::updateAnalysis(Value *Val, ConcreteType Data,
                                  Value *Origin) {
  updateAnalysis(Val, TypeTree(Data), Origin);
}

void TypeAnalyzer::updateAnalysis(Value *Val, BaseType Data, Value *Origin) {
  updateAnalysis(Val, TypeTree(ConcreteType(Data)), Origin);
}

void TypeAnalyzer::addToWorkList(Value *Val) {
  // Only consider instructions/arguments
  if (!isa<Instruction>(Val) && !isa<Argument>(Val) &&
      !isa<ConstantExpr>(Val) && !isa<GlobalVariable>(Val))
    return;

  // Verify this value comes from the function being analyzed
  if (auto I = dyn_cast<Instruction>(Val)) {
    if (fntypeinfo.Function != I->getParent()->getParent())
      return;
    if (notForAnalysis.count(I->getParent()))
      return;
    if (fntypeinfo.Function != I->getParent()->getParent()) {
      llvm::errs() << "function: " << *fntypeinfo.Function << "\n";
      llvm::errs() << "instf: " << *I->getParent()->getParent() << "\n";
      llvm::errs() << "inst: " << *I << "\n";
    }
    assert(fntypeinfo.Function == I->getParent()->getParent());
  } else if (auto Arg = dyn_cast<Argument>(Val)) {
    if (fntypeinfo.Function != Arg->getParent()) {
      llvm::errs() << "fn: " << *fntypeinfo.Function << "\n";
      llvm::errs() << "argparen: " << *Arg->getParent() << "\n";
      llvm::errs() << "val: " << *Arg << "\n";
    }
    assert(fntypeinfo.Function == Arg->getParent());
  }

  // Add to workList
  workList.insert(Val);
}

void TypeAnalyzer::updateAnalysis(Value *Val, TypeTree Data, Value *Origin) {
  if (Val->getType()->isVoidTy())
    return;
  // ConstantData's and Functions don't have analysis updated
  // We don't do "Constant" as globals are "Constant" types
  if (isa<ConstantData>(Val) || isa<Function>(Val)) {
    return;
  }

  if (auto GV = dyn_cast<GlobalVariable>(Val)) {
    if (hasMetadata(GV, "enzyme_ta_norecur"))
      return;
  }

  if (auto CE = dyn_cast<ConstantExpr>(Val)) {
    if (CE->isCast() && isa<ConstantInt>(CE->getOperand(0))) {
      return;
    }
    if (CE->getOpcode() == Instruction::GetElementPtr &&
        isa<ConstantPointerNull>(CE->getOperand(0)))
      return;
  }

  if (auto I = dyn_cast<Instruction>(Val)) {
    if (fntypeinfo.Function != I->getParent()->getParent()) {
      llvm::errs() << "function: " << *fntypeinfo.Function << "\n";
      llvm::errs() << "instf: " << *I->getParent()->getParent() << "\n";
      llvm::errs() << "inst: " << *I << "\n";
    }
    assert(fntypeinfo.Function == I->getParent()->getParent());
    assert(Origin);
    if (!EnzymeStrictAliasing) {
      if (auto OI = dyn_cast<Instruction>(Origin)) {
        if (OI->getParent() != I->getParent() &&
            !PDT.dominates(OI->getParent(), I->getParent())) {
          bool allocationWithAllUsersInBlock = false;
          if (auto AI = dyn_cast<AllocaInst>(I)) {
            allocationWithAllUsersInBlock = true;
            for (auto U : AI->users()) {
              auto P = cast<Instruction>(U)->getParent();
              if (P == OI->getParent())
                continue;
              if (PDT.dominates(OI->getParent(), P))
                continue;
              allocationWithAllUsersInBlock = false;
              break;
            }
          }
          if (!allocationWithAllUsersInBlock) {
            if (EnzymePrintType) {
              llvm::errs() << " skipping update into ";
              I->print(llvm::errs(), *MST);
              llvm::errs() << " of " << Data.str() << " from ";
              OI->print(llvm::errs(), *MST);
              llvm::errs() << "\n";
            }
            return;
          }
        }
      }
    }
  } else if (auto Arg = dyn_cast<Argument>(Val)) {
    assert(fntypeinfo.Function == Arg->getParent());
    if (!EnzymeStrictAliasing)
      if (auto OI = dyn_cast<Instruction>(Origin)) {
        auto I = &*fntypeinfo.Function->getEntryBlock().begin();
        if (OI->getParent() != I->getParent() &&
            !PDT.dominates(OI->getParent(), I->getParent())) {
          if (EnzymePrintType) {
            llvm::errs() << " skipping update into ";
            Arg->print(llvm::errs(), *MST);
            llvm::errs() << " of " << Data.str() << " from ";
            OI->print(llvm::errs(), *MST);
            llvm::errs() << "\n";
          }
          return;
        }
      }
  }

  // Attempt to update the underlying analysis
  bool LegalOr = true;
  if (analysis.find(Val) == analysis.end() && isa<Constant>(Val)) {
    if (!isa<ConstantExpr>(Val) ||
        cast<ConstantExpr>(Val)->getOpcode() != Instruction::GetElementPtr)
      getConstantAnalysis(cast<Constant>(Val), *this, analysis);
  }

  TypeTree prev = analysis[Val];

  auto &DL = fntypeinfo.Function->getParent()->getDataLayout();
  auto RegSize = (DL.getTypeSizeInBits(Val->getType()) + 7) / 8;
  Data.CanonicalizeInPlace(RegSize, DL);
  bool Changed =
      analysis[Val].checkedOrIn(Data, /*PointerIntSame*/ false, LegalOr);

  // Print the update being made, if requested
  if (EnzymePrintType) {
    llvm::errs() << "updating analysis of val: ";
    Val->print(llvm::errs(), *MST);
    llvm::errs() << " current: " << prev.str() << " new " << Data.str();
    if (Origin) {
      llvm::errs() << " from ";
      Origin->print(llvm::errs(), *MST);
    }
    llvm::errs() << " Changed=" << Changed << " legal=" << LegalOr << "\n";
  }

  if (!LegalOr) {
    if (direction != BOTH) {
      Invalid = true;
      return;
    }
    std::string str;
    raw_string_ostream ss(str);
    if (!CustomErrorHandler) {
      llvm::errs() << *fntypeinfo.Function->getParent() << "\n";
      llvm::errs() << *fntypeinfo.Function << "\n";
      dump(ss);
    }
    ss << "Illegal updateAnalysis prev:" << prev.str() << " new: " << Data.str()
       << "\n";
    ss << "val: " << *Val;
    if (Origin)
      ss << " origin=" << *Origin;

    if (CustomErrorHandler) {
      CustomErrorHandler(str.c_str(), wrap(Val), ErrorType::IllegalTypeAnalysis,
                         (void *)this, wrap(Origin), nullptr);
    }
    if (auto I = dyn_cast<Instruction>(Val)) {
      EmitFailure("IllegalUpdateAnalysis", I->getDebugLoc(), I, ss.str());
      exit(1);
    } else if (auto I = dyn_cast_or_null<Instruction>(Origin)) {
      EmitFailure("IllegalUpdateAnalysis", I->getDebugLoc(), I, ss.str());
      exit(1);
    } else {
      llvm::errs() << ss.str() << "\n";
    }
    report_fatal_error("Performed illegal updateAnalysis");
  }

  if (Changed) {

    if (auto GV = dyn_cast<GlobalVariable>(Val)) {
      if (GV->getValueType()->isSized()) {
        auto Size = (DL.getTypeSizeInBits(GV->getValueType()) + 7) / 8;
        Data = analysis[Val].Lookup(Size, DL).Only(-1, nullptr);
        Data.insert({-1}, BaseType::Pointer);
        analysis[Val] = Data;
        Origin = Val;
      }
    }
    // Add val so it can explicitly propagate this new info, if able to
    if (Val != Origin)
      addToWorkList(Val);

    // Add users and operands of the value so they can update from the new
    // operand/use
    for (User *U : Val->users()) {
      if (U != Origin) {

        if (auto I = dyn_cast<Instruction>(U)) {
          if (fntypeinfo.Function != I->getParent()->getParent()) {
            continue;
          }
        }

        addToWorkList(U);

        // per the handling of phi's
        if (auto BO = dyn_cast<BinaryOperator>(U)) {
          for (User *U2 : BO->users()) {
            if (isa<PHINode>(U2) && U2 != Origin) {
              addToWorkList(U2);
            }
          }
        }
      }
    }

    if (User *US = dyn_cast<User>(Val)) {
      for (Value *Op : US->operands()) {
        if (Op != Origin) {
          addToWorkList(Op);
        }
      }
    }
  }
}

/// Analyze type info given by the arguments, possibly adding to work queue
void TypeAnalyzer::prepareArgs() {
  // Propagate input type information for arguments
  for (auto &pair : fntypeinfo.Arguments) {
    assert(pair.first->getParent() == fntypeinfo.Function);
    updateAnalysis(pair.first, pair.second, pair.first);
  }

  // Get type and other information about argument
  // getAnalysis may add more information so this
  // is necessary/useful
  for (Argument &Arg : fntypeinfo.Function->args()) {
    updateAnalysis(&Arg, getAnalysis(&Arg), &Arg);
  }

  // Propagate return value type information
  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (Instruction &I : BB) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(&I)) {
        if (Value *RV = RI->getReturnValue()) {
          updateAnalysis(RV, fntypeinfo.Return, RV);
          updateAnalysis(RV, getAnalysis(RV), RV);
        }
      }
    }
  }
}

/// Analyze type info given by the TBAA, possibly adding to work queue
void TypeAnalyzer::considerTBAA() {
  auto &DL = fntypeinfo.Function->getParent()->getDataLayout();

  for (BasicBlock &BB : *fntypeinfo.Function) {
    if (notForAnalysis.count(&BB))
      continue;
    for (Instruction &I : BB) {
      if (auto MD = I.getMetadata("enzyme_type")) {
        auto TT = TypeTree::fromMD(MD);

        auto RegSize = (DL.getTypeSizeInBits(I.getType()) + 7) / 8;
        for (const auto &pair : TT.getMapping()) {
          if (pair.first[0] != -1) {
            if ((size_t)pair.first[0] >= RegSize) {
              llvm::errs() << " bad enzyme_type " << TT.str()
                           << " RegSize=" << RegSize << " I:" << I << "\n";
              llvm::report_fatal_error("Canonicalization failed");
            }
          }
        }
        updateAnalysis(&I, TT, &I);
      }

      if (CallBase *call = dyn_cast<CallBase>(&I)) {
#if LLVM_VERSION_MAJOR >= 14
        size_t num_args = call->arg_size();
#else
        size_t num_args = call->getNumArgOperands();
#endif

        if (call->getAttributes().hasAttribute(AttributeList::ReturnIndex,
                                               "enzyme_type")) {
          auto attr = call->getAttributes().getAttribute(
              AttributeList::ReturnIndex, "enzyme_type");
          auto TT =
              TypeTree::parse(attr.getValueAsString(), call->getContext());

          auto RegSize = I.getType()->isVoidTy()
                             ? 0
                             : (DL.getTypeSizeInBits(I.getType()) + 7) / 8;
          for (const auto &pair : TT.getMapping()) {
            if (pair.first[0] != -1) {
              if ((size_t)pair.first[0] >= RegSize) {
                llvm::errs() << " bad enzyme_type " << TT.str()
                             << " RegSize=" << RegSize << " I:" << I << "\n";
                llvm::report_fatal_error("Canonicalization failed");
              }
            }
          }
          updateAnalysis(call, TT, call);
        }
        for (size_t i = 0; i < num_args; i++) {
          if (call->getAttributes().hasParamAttr(i, "enzyme_type")) {
            auto attr = call->getAttributes().getParamAttr(i, "enzyme_type");
            auto TT =
                TypeTree::parse(attr.getValueAsString(), call->getContext());
            auto RegSize = I.getType()->isVoidTy()
                               ? 0
                               : (DL.getTypeSizeInBits(I.getType()) + 7) / 8;
            for (const auto &pair : TT.getMapping()) {
              if (pair.first[0] != -1) {
                if ((size_t)pair.first[0] >= RegSize) {
                  llvm::errs() << " bad enzyme_type " << TT.str()
                               << " RegSize=" << RegSize << " I:" << I << "\n";
                  llvm::report_fatal_error("Canonicalization failed");
                }
              }
            }
            updateAnalysis(call->getArgOperand(i), TT, call);
          }
        }

        Function *F = call->getCalledFunction();

        if (F) {
          if (F->getAttributes().hasAttribute(AttributeList::ReturnIndex,
                                              "enzyme_type")) {
            auto attr = F->getAttributes().getAttribute(
                AttributeList::ReturnIndex, "enzyme_type");
            auto TT =
                TypeTree::parse(attr.getValueAsString(), call->getContext());
            auto RegSize = I.getType()->isVoidTy()
                               ? 0
                               : (DL.getTypeSizeInBits(I.getType()) + 7) / 8;
            for (const auto &pair : TT.getMapping()) {
              if (pair.first[0] != -1) {
                if ((size_t)pair.first[0] >= RegSize) {
                  llvm::errs() << " bad enzyme_type " << TT.str()
                               << " RegSize=" << RegSize << " I:" << I << "\n";
                  llvm::report_fatal_error("Canonicalization failed");
                }
              }
            }
            updateAnalysis(call, TT, call);
          }
          size_t f_num_args = F->arg_size();
          for (size_t i = 0; i < f_num_args; i++) {
            if (F->getAttributes().hasParamAttr(i, "enzyme_type")) {
              auto attr = F->getAttributes().getParamAttr(i, "enzyme_type");
              auto TT =
                  TypeTree::parse(attr.getValueAsString(), call->getContext());
              auto RegSize = I.getType()->isVoidTy()
                                 ? 0
                                 : (DL.getTypeSizeInBits(I.getType()) + 7) / 8;
              for (const auto &pair : TT.getMapping()) {
                if (pair.first[0] != -1) {
                  if ((size_t)pair.first[0] >= RegSize) {
                    llvm::errs()
                        << " bad enzyme_type " << TT.str()
                        << " RegSize=" << RegSize << " I:" << I << "\n";
                    llvm::report_fatal_error("Canonicalization failed");
                  }
                }
              }
              updateAnalysis(call->getArgOperand(i), TT, call);
            }
          }
        }

        if (auto castinst = dyn_cast<ConstantExpr>(call->getCalledOperand())) {
          if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
              F = fn;
            }
        }
        if (F && F->getName().contains("__enzyme_float")) {
          assert(num_args == 1 || num_args == 2);
          assert(call->getArgOperand(0)->getType()->isPointerTy());
          TypeTree TT;
          ssize_t num = 1;
          if (num_args == 2) {
            assert(isa<ConstantInt>(call->getArgOperand(1)));
            auto CI = cast<ConstantInt>(call->getArgOperand(1));
            if (CI->isNegative())
              num = -1;
            else
              num = CI->getLimitedValue();
          }
          if (num == -1)
            TT.insert({(int)num}, Type::getFloatTy(call->getContext()));
          else
            for (size_t i = 0; i < (size_t)num; i += 4)
              TT.insert({(int)i}, Type::getFloatTy(call->getContext()));
          TT.insert({}, BaseType::Pointer);
          updateAnalysis(call->getOperand(0), TT.Only(-1, call), call);
        }
        if (F && F->getName().contains("__enzyme_double")) {
          assert(num_args == 1 || num_args == 2);
          assert(call->getArgOperand(0)->getType()->isPointerTy());
          TypeTree TT;
          size_t num = 1;
          if (num_args == 2) {
            assert(isa<ConstantInt>(call->getArgOperand(1)));
            num = cast<ConstantInt>(call->getArgOperand(1))->getLimitedValue();
          }
          for (size_t i = 0; i < num; i += 8)
            TT.insert({(int)i}, Type::getDoubleTy(call->getContext()));
          TT.insert({}, BaseType::Pointer);
          updateAnalysis(call->getOperand(0), TT.Only(-1, call), call);
        }
        if (F && F->getName().contains("__enzyme_integer")) {
          assert(num_args == 1 || num_args == 2);
          assert(call->getArgOperand(0)->getType()->isPointerTy());
          size_t num = 1;
          if (num_args == 2) {
            assert(isa<ConstantInt>(call->getArgOperand(1)));
            num = cast<ConstantInt>(call->getArgOperand(1))->getLimitedValue();
          }
          TypeTree TT;
          for (size_t i = 0; i < num; i++)
            TT.insert({(int)i}, BaseType::Integer);
          TT.insert({}, BaseType::Pointer);
          updateAnalysis(call->getOperand(0), TT.Only(-1, call), call);
        }
        if (F && F->getName().contains("__enzyme_pointer")) {
          assert(num_args == 1 || num_args == 2);
          assert(call->getArgOperand(0)->getType()->isPointerTy());
          TypeTree TT;
          size_t num = 1;
          if (num_args == 2) {
            assert(isa<ConstantInt>(call->getArgOperand(1)));
            num = cast<ConstantInt>(call->getArgOperand(1))->getLimitedValue();
          }
          for (size_t i = 0; i < num;
               i += ((DL.getPointerSizeInBits() + 7) / 8))
            TT.insert({(int)i}, BaseType::Pointer);
          TT.insert({}, BaseType::Pointer);
          updateAnalysis(call->getOperand(0), TT.Only(-1, call), call);
        }
        if (F) {
          StringSet<> JuliaKnownTypes = {"julia.gc_alloc_obj",
                                         "jl_alloc_array_1d",
                                         "jl_alloc_array_2d",
                                         "jl_alloc_array_3d",
                                         "ijl_alloc_array_1d",
                                         "ijl_alloc_array_2d",
                                         "ijl_alloc_array_3d",
                                         "jl_gc_alloc_typed",
                                         "ijl_gc_alloc_typed",
                                         "jl_alloc_genericmemory",
                                         "ijl_alloc_genericmemory",
                                         "jl_alloc_genericmemory_unchecked",
                                         "ijl_alloc_genericmemory_unchecked",
                                         "jl_new_array",
                                         "ijl_new_array"};
          if (JuliaKnownTypes.count(F->getName())) {
            visitCallBase(*call);
            continue;
          }
        }
      }

      TypeTree vdptr = parseTBAA(I, DL, MST);

      // If we don't have any useful information,
      // don't bother updating
      if (!vdptr.isKnownPastPointer())
        continue;

      if (CallBase *call = dyn_cast<CallBase>(&I)) {
        if (call->getCalledFunction() &&
            (call->getCalledFunction()->getIntrinsicID() == Intrinsic::memcpy ||
             call->getCalledFunction()->getIntrinsicID() ==
                 Intrinsic::memmove)) {
          int64_t copySize = 1;
          for (auto val : fntypeinfo.knownIntegralValues(call->getOperand(2),
                                                         DT, intseen, SE)) {
            copySize = max(copySize, val);
          }
          TypeTree update =
              vdptr
                  .ShiftIndices(DL, /*init offset*/ 0,
                                /*max size*/ copySize, /*new offset*/ 0)
                  .Only(-1, call);

          updateAnalysis(call->getOperand(0), update, call);
          updateAnalysis(call->getOperand(1), update, call);
          continue;
        } else if (call->getCalledFunction() &&
                   (call->getCalledFunction()->getIntrinsicID() ==
                        Intrinsic::memset ||
                    call->getCalledFunction()->getName() ==
                        "memset_pattern16")) {
          int64_t copySize = 1;
          for (auto val : fntypeinfo.knownIntegralValues(call->getOperand(2),
                                                         DT, intseen, SE)) {
            copySize = max(copySize, val);
          }
          TypeTree update =
              vdptr
                  .ShiftIndices(DL, /*init offset*/ 0,
                                /*max size*/ copySize, /*new offset*/ 0)
                  .Only(-1, call);

          updateAnalysis(call->getOperand(0), update, call);
          continue;
#if LLVM_VERSION_MAJOR >= 20
        } else if (call->getCalledFunction() &&
                   (call->getCalledFunction()->getIntrinsicID() ==
                    Intrinsic::experimental_memset_pattern)) {
          int64_t copySize = 1;
          for (auto val : fntypeinfo.knownIntegralValues(call->getOperand(2),
                                                         DT, intseen, SE)) {
            copySize = max(copySize, val);
          }
          TypeTree update =
              vdptr
                  .ShiftIndices(DL, /*init offset*/ 0,
                                /*max size*/ copySize, /*new offset*/ 0)
                  .Only(-1, call);

          updateAnalysis(call->getOperand(0), update, call);
          continue;
#endif
        } else if (call->getCalledFunction() &&
                   call->getCalledFunction()->getIntrinsicID() ==
                       Intrinsic::masked_gather) {
          auto VT = cast<VectorType>(call->getType());
          auto LoadSize = (DL.getTypeSizeInBits(VT) + 7) / 8;
          TypeTree req = vdptr.Only(-1, call);
          updateAnalysis(call, req.Lookup(LoadSize, DL), call);
          // TODO use mask to propagate up to relevant pointer
        } else if (call->getCalledFunction() &&
                   call->getCalledFunction()->getIntrinsicID() ==
                       Intrinsic::masked_scatter) {
          // TODO use mask to propagate up to relevant pointer
        } else if (call->getCalledFunction() &&
                   call->getCalledFunction()->getIntrinsicID() ==
                       Intrinsic::masked_load) {
          auto VT = cast<VectorType>(call->getType());
          auto LoadSize = (DL.getTypeSizeInBits(VT) + 7) / 8;
          TypeTree req = vdptr.Only(-1, call);
          updateAnalysis(call, req.Lookup(LoadSize, DL), call);
          // TODO use mask to propagate up to relevant pointer
        } else if (call->getCalledFunction() &&
                   call->getCalledFunction()->getIntrinsicID() ==
                       Intrinsic::masked_store) {
          // TODO use mask to propagate up to relevant pointer
        } else if (call->getType()->isPointerTy()) {
          updateAnalysis(call, vdptr.Only(-1, call), call);
        } else {
          llvm::errs() << " unknown tbaa call instruction user inst: " << I
                       << " vdptr: " << vdptr.str() << "\n";
        }
      } else if (auto SI = dyn_cast<StoreInst>(&I)) {
        auto StoreSize =
            (DL.getTypeSizeInBits(SI->getValueOperand()->getType()) + 7) / 8;
        updateAnalysis(SI->getPointerOperand(),
                       vdptr
                           // Don't propagate "Anything" into ptr
                           .PurgeAnything()
                           // Cut off any values outside of store
                           .ShiftIndices(DL, /*init offset*/ 0,
                                         /*max size*/ StoreSize,
                                         /*new offset*/ 0)
                           .Only(-1, SI),
                       SI);
        TypeTree req = vdptr.Only(-1, SI);
        updateAnalysis(SI->getValueOperand(), req.Lookup(StoreSize, DL), SI);
      } else if (auto LI = dyn_cast<LoadInst>(&I)) {
        auto LoadSize = (DL.getTypeSizeInBits(LI->getType()) + 7) / 8;
        updateAnalysis(LI->getPointerOperand(),
                       vdptr
                           // Don't propagate "Anything" into ptr
                           .PurgeAnything()
                           // Cut off any values outside of load
                           .ShiftIndices(DL, /*init offset*/ 0,
                                         /*max size*/ LoadSize,
                                         /*new offset*/ 0)
                           .Only(-1, LI),
                       LI);
        TypeTree req = vdptr.Only(-1, LI);
        updateAnalysis(LI, req.Lookup(LoadSize, DL), LI);
      } else {
        llvm::errs() << " inst: " << I << " vdptr: " << vdptr.str() << "\n";
        assert(0 && "unknown tbaa instruction user");
        llvm_unreachable("unknown tbaa instruction user");
      }
    }
  }
}

void TypeAnalyzer::runPHIHypotheses() {
  if (PHIRecur)
    return;
  bool Changed;
  do {
    Changed = false;
    for (BasicBlock &BB : *fntypeinfo.Function) {
      for (Instruction &inst : BB) {
        if (PHINode *phi = dyn_cast<PHINode>(&inst)) {
          if (direction & DOWN && phi->getType()->isIntOrIntVectorTy() &&
              !getAnalysis(phi).isKnown()) {
            // Assume that this is an integer, does that mean we can prove that
            // the incoming operands are integral

            TypeAnalyzer tmpAnalysis(fntypeinfo, interprocedural,
                                     notForAnalysis, *this, DOWN,
                                     /*PHIRecur*/ true);
            tmpAnalysis.intseen = intseen;
            tmpAnalysis.analysis = analysis;
            tmpAnalysis.analysis[phi] =
                TypeTree(BaseType::Integer).Only(-1, phi);
            for (auto U : phi->users()) {
              if (auto I = dyn_cast<Instruction>(U)) {
                tmpAnalysis.visit(*I);
              }
            }
            tmpAnalysis.run();
            if (!tmpAnalysis.Invalid) {
              TypeTree Result = tmpAnalysis.getAnalysis(phi);
              for (auto &op : phi->incoming_values()) {
                Result &= tmpAnalysis.getAnalysis(op);
              }
              if (Result == TypeTree(BaseType::Integer).Only(-1, phi) ||
                  Result == TypeTree(BaseType::Anything).Only(-1, phi)) {
                updateAnalysis(phi, Result, phi);
                for (auto &pair : tmpAnalysis.analysis) {
                  updateAnalysis(pair.first, pair.second, phi);
                }
                Changed = true;
              }
            }
          }

          if (direction & DOWN && phi->getType()->isFPOrFPVectorTy() &&
              !getAnalysis(phi).isKnown()) {
            // Assume that this is an integer, does that mean we can prove that
            // the incoming operands are integral
            TypeAnalyzer tmpAnalysis(fntypeinfo, interprocedural,
                                     notForAnalysis, *this, DOWN,
                                     /*PHIRecur*/ true);
            tmpAnalysis.intseen = intseen;
            tmpAnalysis.analysis = analysis;
            tmpAnalysis.analysis[phi] =
                TypeTree(phi->getType()->getScalarType()).Only(-1, phi);
            for (auto U : phi->users()) {
              if (auto I = dyn_cast<Instruction>(U)) {
                tmpAnalysis.visit(*I);
              }
            }
            tmpAnalysis.run();
            if (!tmpAnalysis.Invalid) {
              TypeTree Result = tmpAnalysis.getAnalysis(phi);
              for (auto &op : phi->incoming_values()) {
                Result &= tmpAnalysis.getAnalysis(op);
              }
              if (Result ==
                      TypeTree(phi->getType()->getScalarType()).Only(-1, phi) ||
                  Result == TypeTree(BaseType::Anything).Only(-1, phi)) {
                updateAnalysis(phi, Result, phi);
                for (auto &pair : tmpAnalysis.analysis) {
                  updateAnalysis(pair.first, pair.second, phi);
                }
                Changed = true;
              }
            }
          }
        }
      }
    }
  } while (Changed);
  return;
}

void TypeAnalyzer::run() {

  TimeTraceScope timeScope("Type Analysis", fntypeinfo.Function->getName());

  // This function runs a full round of type analysis.
  // This works by doing two stages of analysis,
  // with a "deduced integer types for unused" values
  // sandwiched in-between. This is done because we only
  // perform that check for values without types.
  //
  // For performance reasons in each round of type analysis
  // only analyze any call instances after all other potential
  // updates have been done. This is to minimize the number
  // of expensive interprocedural analyses
  std::deque<CallBase *> pendingCalls;

  do {
    while (!Invalid && workList.size()) {
      auto todo = *workList.begin();
      workList.erase(workList.begin());
      if (auto call = dyn_cast<CallBase>(todo)) {
        StringRef funcName = getFuncNameFromCall(call);
        auto ci = getFunctionFromCall(call);
        if (ci && !ci->empty()) {
          if (interprocedural.CustomRules.find(funcName) ==
              interprocedural.CustomRules.end()) {
            pendingCalls.push_back(call);
            continue;
          }
        }
      }
      visitValue(*todo);
    }

    if (pendingCalls.size() > 0) {
      auto todo = pendingCalls.front();
      pendingCalls.pop_front();
      visitValue(*todo);
      continue;
    } else
      break;

  } while (1);

  runPHIHypotheses();

  do {

    while (!Invalid && workList.size()) {
      auto todo = *workList.begin();
      workList.erase(workList.begin());
      if (auto ci = dyn_cast<CallBase>(todo)) {
        pendingCalls.push_back(ci);
        continue;
      }
      visitValue(*todo);
    }

    if (pendingCalls.size() > 0) {
      auto todo = pendingCalls.front();
      pendingCalls.pop_front();
      visitValue(*todo);
      continue;
    } else
      break;

  } while (1);
}

void TypeAnalyzer::visitValue(Value &val) {
  if (auto CE = dyn_cast<ConstantExpr>(&val)) {
    visitConstantExpr(*CE);
  }

  if (isa<Constant>(&val)) {
    return;
  }

  if (!isa<Argument>(&val) && !isa<Instruction>(&val))
    return;

  if (auto *FPMO = dyn_cast<FPMathOperator>(&val)) {
    if (FPMO->getOpcode() == Instruction::FNeg) {
      Value *op = FPMO->getOperand(0);
      auto ty = op->getType()->getScalarType();
      assert(ty->isFloatingPointTy());
      ConcreteType dt(ty);
      updateAnalysis(op, TypeTree(ty).Only(-1, nullptr),
                     cast<Instruction>(&val));
      updateAnalysis(FPMO, TypeTree(ty).Only(-1, nullptr),
                     cast<Instruction>(&val));
      return;
    }
  }

  if (auto inst = dyn_cast<Instruction>(&val)) {
    visit(*inst);
  }
}

void TypeAnalyzer::visitConstantExpr(ConstantExpr &CE) {
  if (CE.isCast()) {
    if (direction & DOWN)
      updateAnalysis(&CE, getAnalysis(CE.getOperand(0)), &CE);
    if (direction & UP)
      updateAnalysis(CE.getOperand(0), getAnalysis(&CE), &CE);
    return;
  }
  if (CE.getOpcode() == Instruction::GetElementPtr) {
    visitGEPOperator(*cast<GEPOperator>(&CE));
    return;
  }
  auto I = CE.getAsInstruction();
  I->insertBefore(fntypeinfo.Function->getEntryBlock().getTerminator());
  analysis[I] = analysis[&CE];
  visit(*I);
  updateAnalysis(&CE, analysis[I], &CE);
  analysis.erase(I);
  if (workList.remove(I)) {
    workList.insert(&CE);
  }
  I->eraseFromParent();
}

void TypeAnalyzer::visitCmpInst(CmpInst &cmp) {
  // No directionality check needed as always true
  updateAnalysis(&cmp, TypeTree(BaseType::Integer).Only(-1, &cmp), &cmp);
  if (direction & UP) {
    updateAnalysis(
        cmp.getOperand(0),
        TypeTree(getAnalysis(cmp.getOperand(1)).Inner0().PurgeAnything())
            .Only(-1, &cmp),
        &cmp);
    updateAnalysis(
        cmp.getOperand(1),
        TypeTree(getAnalysis(cmp.getOperand(0)).Inner0().PurgeAnything())
            .Only(-1, &cmp),
        &cmp);
  }
}

void TypeAnalyzer::visitAllocaInst(AllocaInst &I) {
  // No directionality check needed as always true
  updateAnalysis(I.getArraySize(), TypeTree(BaseType::Integer).Only(-1, &I),
                 &I);

  auto ptr = TypeTree(BaseType::Pointer);

  if (auto CI = dyn_cast<ConstantInt>(I.getArraySize())) {
    auto &DL = I.getParent()->getParent()->getParent()->getDataLayout();
    auto LoadSize = CI->getZExtValue() *
                    (DL.getTypeSizeInBits(I.getAllocatedType()) + 7) / 8;
    // Only propagate mappings in range that aren't "Anything" into the pointer
    ptr |= getAnalysis(&I).Lookup(LoadSize, DL);
  }
  updateAnalysis(&I, ptr.Only(-1, &I), &I);
}

void TypeAnalyzer::visitLoadInst(LoadInst &I) {
  auto &DL = I.getParent()->getParent()->getParent()->getDataLayout();
  auto LoadSize = (DL.getTypeSizeInBits(I.getType()) + 7) / 8;

  if (direction & UP) {
    // Only propagate mappings in range that aren't "Anything" into the pointer
    auto ptr = getAnalysis(&I).PurgeAnything().ShiftIndices(
        DL, /*start*/ 0, LoadSize, /*addOffset*/ 0);
    ptr |= TypeTree(BaseType::Pointer);
    updateAnalysis(I.getOperand(0), ptr.Only(-1, &I), &I);
  }
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)).Lookup(LoadSize, DL), &I);
}

void TypeAnalyzer::visitStoreInst(StoreInst &I) {
  auto &DL = I.getParent()->getParent()->getParent()->getDataLayout();
  auto StoreSize =
      (DL.getTypeSizeInBits(I.getValueOperand()->getType()) + 7) / 8;

  // Rust specific rule, if storing an integer equal to the alignment
  // of a store, assuming nothing (or assume it is a pointer)
  // https://doc.rust-lang.org/src/core/ptr/non_null.rs.html#70-78
  if (RustTypeRules)
    if (auto CI = dyn_cast<ConstantInt>(I.getValueOperand())) {
      auto alignment = I.getAlign().value();

      if (CI->getLimitedValue() == alignment) {
        return;
      }
    }

  // Only propagate mappings in range that aren't "Anything" into the pointer
  auto ptr = TypeTree(BaseType::Pointer);
  auto purged = getAnalysis(I.getValueOperand())
                    .PurgeAnything()
                    .ShiftIndices(DL, /*start*/ 0, StoreSize, /*addOffset*/ 0)
                    .ReplaceMinus();
  ptr |= purged;

  if (direction & UP) {
    updateAnalysis(I.getPointerOperand(), ptr.Only(-1, &I), &I);

    // Note that we also must purge anything from ptr => value in case we store
    // to a nullptr which has type [-1, -1]: Anything. While storing to a
    // nullptr is obviously bad, this doesn't mean the value we're storing is an
    // Anything
    updateAnalysis(I.getValueOperand(),
                   getAnalysis(I.getPointerOperand())
                       .PurgeAnything()
                       .Lookup(StoreSize, DL),
                   &I);
  }
}

// Give a list of sets representing the legal set of values at a given index
// return a set of all possible combinations of those values
template <typename T>
std::set<SmallVector<T, 4>> getSet(ArrayRef<std::set<T>> todo, size_t idx) {
  assert(idx < todo.size());
  std::set<SmallVector<T, 4>> out;
  if (idx == 0) {
    for (auto val : todo[0]) {
      out.insert({val});
    }
    return out;
  }

  auto old = getSet(todo, idx - 1);
  for (const auto &oldv : old) {
    for (auto val : todo[idx]) {
      auto nex = oldv;
      nex.push_back(val);
      out.insert(nex);
    }
  }
  return out;
}

void TypeAnalyzer::visitGetElementPtrInst(GetElementPtrInst &gep) {
  visitGEPOperator(*cast<GEPOperator>(&gep));
}

void TypeAnalyzer::visitGEPOperator(GEPOperator &gep) {
  auto inst = dyn_cast<Instruction>(&gep);
  if (isa<UndefValue>(gep.getPointerOperand())) {
    updateAnalysis(&gep, TypeTree(BaseType::Anything).Only(-1, inst), &gep);
    return;
  }
  if (isa<ConstantPointerNull>(gep.getPointerOperand())) {
    bool nonZero = false;
    bool legal = true;
    for (auto I = gep.idx_begin(), E = gep.idx_end(); I != E; I++) {
      auto ind = I->get();
      if (auto CI = dyn_cast<ConstantInt>(ind)) {
        if (!CI->isZero()) {
          nonZero = true;
          continue;
        }
      }
      auto CT = getAnalysis(ind).Inner0();
      if (CT == BaseType::Integer) {
        continue;
      }
      legal = false;
      break;
    }
    if (legal && nonZero) {
      updateAnalysis(&gep, TypeTree(BaseType::Integer).Only(-1, inst), &gep);
      return;
    }
  }
  if (auto GV = dyn_cast<GlobalVariable>(gep.getPointerOperand())) {
    // from julia code, do not propagate int to operands
    if (GV->getName() == "small_typeof" || GV->getName() == "jl_small_typeof") {
      TypeTree T;
      T.insert({-1}, BaseType::Pointer);
      T.insert({-1, -1}, BaseType::Pointer);
      updateAnalysis(&gep, T, &gep);
      return;
    }
  }

  if (gep.idx_begin() == gep.idx_end()) {
    if (direction & DOWN)
      updateAnalysis(&gep, getAnalysis(gep.getPointerOperand()), &gep);
    if (direction & UP)
      updateAnalysis(gep.getPointerOperand(), getAnalysis(&gep), &gep);
    return;
  }

  auto &DL = fntypeinfo.Function->getParent()->getDataLayout();

  auto pointerAnalysis = getAnalysis(gep.getPointerOperand());

  // If we know that the pointer operand is indeed a pointer, then the indicies
  // must be integers Note that we can't do this if we don't know the pointer
  // operand is a pointer since doing 1[pointer] is legal
  //  sadly this still may not work since (nullptr)[fn] => fn where fn is
  //  pointer and not int (whereas nullptr is a pointer) However if we are
  //  inbounds you are only allowed to have nullptr[0] or nullptr[nullptr],
  //  making this valid
  // Assuming nullptr[nullptr] doesn't occur in practice, the following
  // is valid. We could make it always valid by checking the pointer
  // operand explicitly is a pointer.
  if (direction & UP) {
    bool has_non_const_idx = false;
    for (auto I = gep.idx_begin(), E = gep.idx_end(); I != E; I++) {
      auto ind = I->get();
      if (!isa<ConstantInt>(ind)) {
        has_non_const_idx = true;
        break;
      }
    }

    if (has_non_const_idx &&
        (gep.isInBounds() ||
         (!EnzymeStrictAliasing &&
          pointerAnalysis.Inner0() == BaseType::Pointer &&
          getAnalysis(&gep).Inner0() == BaseType::Pointer))) {
      for (auto I = gep.idx_begin(), E = gep.idx_end(); I != E; I++) {
        auto ind = I->get();
        updateAnalysis(ind, TypeTree(BaseType::Integer).Only(-1, inst), &gep);
      }
    }
  }

  // If one of these is known to be a pointer, propagate it if either in bounds
  // or all operands are integral/unknown
  bool pointerPropagate = gep.isInBounds();
  if (!pointerPropagate) {
    bool allIntegral = true;
    for (auto I = gep.idx_begin(), E = gep.idx_end(); I != E; I++) {
      auto ind = I->get();
      auto CT = getAnalysis(ind).Inner0();
      if (CT != BaseType::Integer && CT != BaseType::Anything) {
        allIntegral = false;
        break;
      }
    }
    if (allIntegral)
      pointerPropagate = true;
  }

  if (!pointerPropagate)
    return;

  if (direction & DOWN) {
    bool legal = true;
    auto keepMinus = pointerAnalysis.KeepMinusOne(legal);
    if (!legal) {
      if (CustomErrorHandler)
        CustomErrorHandler("Could not keep minus one", wrap(&gep),
                           ErrorType::IllegalTypeAnalysis, this, nullptr,
                           nullptr);
      else {
        dump();
        llvm::errs() << " could not perform minus one for gep'd: " << gep
                     << "\n";
      }
    }
    updateAnalysis(&gep, keepMinus, &gep);
    // Don't propagate pointer type when the input pointer is null
    if (!isa<ConstantPointerNull>(gep.getPointerOperand())) {
      updateAnalysis(&gep, TypeTree(pointerAnalysis.Inner0()).Only(-1, inst),
                     &gep);
    }
  }
  if (direction & UP)
    updateAnalysis(gep.getPointerOperand(),
                   TypeTree(getAnalysis(&gep).Inner0()).Only(-1, inst), &gep);

  TypeTree upTree;
  TypeTree downTree;

  TypeTree gepData0;
  TypeTree pointerData0;
  if (direction & UP)
    gepData0 = getAnalysis(&gep).Data0();
  if (direction & DOWN)
    pointerData0 = pointerAnalysis.Data0();

  auto BitWidth = DL.getIndexSizeInBits(gep.getPointerAddressSpace());

  APInt constOffset(BitWidth, 0);

#if LLVM_VERSION_MAJOR >= 20
  SmallMapVector<Value *, APInt, 4> VariableOffsets;
#else
  MapVector<Value *, APInt> VariableOffsets;
#endif
  bool legalOffset =
      collectOffset(&gep, DL, BitWidth, VariableOffsets, constOffset);
  (void)legalOffset;
  assert(legalOffset);

  SmallVector<std::set<int>, 4> idnext;

  SmallPtrSet<BasicBlock *, 1> previousLoopInductionHeaders;
  {
    Value *ptr = gep.getPointerOperand();
    while (true) {
      if (auto gepop = dyn_cast<GEPOperator>(ptr)) {
        for (auto I = gepop->idx_begin(), E = gepop->idx_end(); I != E; I++) {
          SmallPtrSet<PHINode *, 1> seen;
          for (auto loopInd : findLoopIndices(*I, LI, DT, seen)) {
            previousLoopInductionHeaders.insert(loopInd);
          }
        }
        ptr = gepop->getPointerOperand();
        continue;
      }
      if (auto CI = dyn_cast<CastInst>(ptr)) {
        ptr = CI->getOperand(0);
        continue;
      }
      break;
    }
  }

  for (auto &pair : VariableOffsets) {
    auto a = pair.first;
    auto iset = fntypeinfo.knownIntegralValues(a, DT, intseen, SE);
    std::set<int> vset;
    for (auto i : iset) {
      // Don't consider negative indices of gep
      if (i < 0)
        continue;
      vset.insert(i);
    }
    if (vset.size() == 0)
      return;

    // If seen the same variable before with > 1 option, we will accidentally
    // do an offset for [option1, option2] * oldOffset + [option1, option2] *
    // newOffset
    //   instead of [option1, option2] * (oldOffset + newOffset).
    //   In this case abort
    //   TODO, in the future, mutually compute the offset together.
    if (vset.size() != 1) {
      SmallPtrSet<PHINode *, 1> seen;
      for (auto loopInd : findLoopIndices(pair.first, LI, DT, seen))
        if (previousLoopInductionHeaders.count(loopInd))
          return;
    }
    idnext.push_back(vset);
  }

  // Stores pair ([whether first offset is zero], offset)
  std::vector<std::pair<bool, int>> offsets;
  Value *firstIdx = *gep.idx_begin();
  if (VariableOffsets.size() == 0) {
    bool firstIsZero = cast<ConstantInt>(firstIdx)->getLimitedValue() == 0;
    offsets.emplace_back(firstIsZero, (int)constOffset.getLimitedValue());
  } else {
    bool firstIsZero = false;
    if (auto CI = dyn_cast<ConstantInt>(firstIdx))
      firstIsZero = CI->getLimitedValue() == 0;
    for (auto vec : getSet<int>(idnext, idnext.size() - 1)) {
      APInt nextOffset = constOffset;
      for (auto [varpair, const_value] : llvm::zip(VariableOffsets, vec)) {
        nextOffset += varpair.second * const_value;
        if (varpair.first == firstIdx)
          firstIsZero = const_value == 0;
      }
      offsets.emplace_back(firstIsZero, (int)nextOffset.getLimitedValue());
    }
  }

  bool seenIdx = false;

  for (auto [firstIsZero, off] : offsets) {
    // TODO also allow negative offsets
    if (off < 0)
      continue;

    int maxSize = -1;
    if (firstIsZero) {
      maxSize = DL.getTypeAllocSizeInBits(gep.getResultElementType()) / 8;
    }

    if (direction & DOWN) {
      auto shft =
          pointerData0.ShiftIndices(DL, /*init offset*/ off,
                                    /*max size*/ maxSize, /*newoffset*/ 0);
      if (seenIdx)
        downTree &= shft;
      else
        downTree = shft;
    }

    if (direction & UP) {
      auto shft = gepData0.ShiftIndices(DL, /*init offset*/ 0, /*max size*/ -1,
                                        /*new offset*/ off);
      if (seenIdx)
        upTree |= shft;
      else
        upTree = shft;
    }
    seenIdx = true;
  }
  if (direction & DOWN)
    updateAnalysis(&gep, downTree.Only(-1, inst), &gep);
  if (direction & UP)
    updateAnalysis(gep.getPointerOperand(), upTree.Only(-1, inst), &gep);
}

void TypeAnalyzer::visitPHINode(PHINode &phi) {
  if (direction & UP) {
    TypeTree upVal = getAnalysis(&phi);
    // only propagate anything's up if there is one
    // incoming value
    Value *seen = phi.getIncomingValue(0);
    for (size_t i = 0, end = phi.getNumIncomingValues(); i < end; ++i) {
      if (seen != phi.getIncomingValue(i)) {
        seen = nullptr;
        break;
      }
    }

    if (!seen) {
      upVal = upVal.PurgeAnything();
    }

    if (EnzymeStrictAliasing || seen) {
      auto L = LI.getLoopFor(phi.getParent());
      bool isHeader = L && L->getHeader() == phi.getParent();
      for (size_t i = 0, end = phi.getNumIncomingValues(); i < end; ++i) {
        if (!isHeader || !L->contains(phi.getIncomingBlock(i))) {
          updateAnalysis(phi.getIncomingValue(i), upVal, &phi);
        }
      }
    } else {
      if (EnzymePrintType) {
        for (size_t i = 0, end = phi.getNumIncomingValues(); i < end; ++i) {
          llvm::errs() << " skipping update into ";
          phi.getIncomingValue(i)->print(llvm::errs(), *MST);
          llvm::errs() << " of " << upVal.str() << " from ";
          phi.print(llvm::errs(), *MST);
          llvm::errs() << "\n";
        }
      }
    }
  }

  assert(phi.getNumIncomingValues() > 0);

  // TODO generalize this (and for recursive, etc)

  for (int i = 0; i < 2; i++) {

    std::deque<Value *> vals;
    std::set<Value *> seen{&phi};
    for (auto &op : phi.incoming_values()) {
      vals.push_back(op);
    }
    SmallVector<BinaryOperator *, 4> bos;

    // Unique values that propagate into this phi
    SmallVector<Value *, 4> UniqueValues;

    while (vals.size()) {
      Value *todo = vals.front();
      vals.pop_front();

      if (auto bo = dyn_cast<BinaryOperator>(todo)) {
        if (bo->getOpcode() == BinaryOperator::Add) {
          if (isa<Constant>(bo->getOperand(0))) {
            bos.push_back(bo);
            todo = bo->getOperand(1);
          }
          if (isa<Constant>(bo->getOperand(1))) {
            bos.push_back(bo);
            todo = bo->getOperand(0);
          }
        }
      }

      if (seen.count(todo))
        continue;
      seen.insert(todo);

      if (auto nphi = dyn_cast<PHINode>(todo)) {
        if (i == 0) {
          for (auto &op : nphi->incoming_values()) {
            vals.push_back(op);
          }
          continue;
        }
      }
      if (auto sel = dyn_cast<SelectInst>(todo)) {
        vals.push_back(sel->getOperand(1));
        vals.push_back(sel->getOperand(2));
        continue;
      }
      UniqueValues.push_back(todo);
    }

    TypeTree PhiTypes;
    bool set = false;

    for (size_t i = 0, size = UniqueValues.size(); i < size; ++i) {
      TypeTree newData = getAnalysis(UniqueValues[i]);
      if (UniqueValues.size() == 2) {
        if (auto BO = dyn_cast<BinaryOperator>(UniqueValues[i])) {
          if (BO->getOpcode() == BinaryOperator::Add ||
              BO->getOpcode() == BinaryOperator::Mul) {
            TypeTree otherData = getAnalysis(UniqueValues[1 - i]);
            // If we are adding/muling to a constant to derive this, we can
            // assume it to be an integer rather than Anything
            if (isa<Constant>(UniqueValues[1 - i])) {
              otherData = TypeTree(BaseType::Integer).Only(-1, &phi);
            }
            if (BO->getOperand(0) == &phi) {
              set = true;
              PhiTypes = otherData;
              bool Legal = true;
              PhiTypes.binopIn(Legal, getAnalysis(BO->getOperand(1)),
                               BO->getOpcode());
              if (!Legal) {
                std::string str;
                raw_string_ostream ss(str);
                if (!CustomErrorHandler) {
                  llvm::errs() << *fntypeinfo.Function->getParent() << "\n";
                  llvm::errs() << *fntypeinfo.Function << "\n";
                  dump(ss);
                }
                ss << "Illegal updateBinop Analysis " << *BO << "\n";
                ss << "Illegal binopIn(0): " << *BO
                   << " lhs: " << PhiTypes.str()
                   << " rhs: " << getAnalysis(BO->getOperand(0)).str() << "\n";
                if (CustomErrorHandler) {
                  CustomErrorHandler(str.c_str(), wrap(BO),
                                     ErrorType::IllegalTypeAnalysis,
                                     (void *)this, wrap(BO), nullptr);
                }
                EmitFailure("IllegalUpdateAnalysis", BO->getDebugLoc(), BO,
                            ss.str());
                report_fatal_error("Performed illegal updateAnalysis");
              }
              break;
            } else if (BO->getOperand(1) == &phi) {
              set = true;
              PhiTypes = getAnalysis(BO->getOperand(0));
              bool Legal = true;
              PhiTypes.binopIn(Legal, otherData, BO->getOpcode());
              if (!Legal) {
                std::string str;
                raw_string_ostream ss(str);
                if (!CustomErrorHandler) {
                  llvm::errs() << *fntypeinfo.Function->getParent() << "\n";
                  llvm::errs() << *fntypeinfo.Function << "\n";
                  dump(ss);
                }
                ss << "Illegal updateBinop Analysis " << *BO << "\n";
                ss << "Illegal binopIn(1): " << *BO
                   << " lhs: " << PhiTypes.str() << " rhs: " << otherData.str()
                   << "\n";
                if (CustomErrorHandler) {
                  CustomErrorHandler(str.c_str(), wrap(BO),
                                     ErrorType::IllegalTypeAnalysis,
                                     (void *)this, wrap(BO), nullptr);
                }
                EmitFailure("IllegalUpdateAnalysis", BO->getDebugLoc(), BO,
                            ss.str());
                report_fatal_error("Performed illegal updateAnalysis");
              }
              break;
            }
          } else if (BO->getOpcode() == BinaryOperator::Sub) {
            // Repeated subtraction from a type X yields the type X back
            TypeTree otherData = getAnalysis(UniqueValues[1 - i]);
            // If we are subtracting from a constant to derive this, we can
            // assume it to be an integer rather than Anything
            if (isa<Constant>(UniqueValues[1 - i])) {
              otherData = TypeTree(BaseType::Integer).Only(-1, &phi);
            }
            if (BO->getOperand(0) == &phi) {
              set = true;
              PhiTypes = otherData;
              break;
            }
          }
        }
      }
      if (set) {
        PhiTypes &= newData;
        // TODO consider the or of anything (see selectinst)
        // however, this cannot be done yet for risk of turning
        // phi's that add floats into anything
        // PhiTypes |= newData.JustAnything();
      } else {
        set = true;
        PhiTypes = newData;
      }
    }

    assert(set);
    // If we are only add / sub / etc to derive a value based off 0
    // we can start by assuming the type of 0 is integer rather
    // than assuming it could be anything (per null)
    if (bos.size() > 0 && UniqueValues.size() == 1 &&
        isa<ConstantInt>(UniqueValues[0]) &&
        (cast<ConstantInt>(UniqueValues[0])->isZero() ||
         cast<ConstantInt>(UniqueValues[0])->isOne())) {
      PhiTypes = TypeTree(BaseType::Integer).Only(-1, &phi);
    }
    for (BinaryOperator *bo : bos) {
      TypeTree vd1 = isa<Constant>(bo->getOperand(0))
                         ? getAnalysis(bo->getOperand(0)).Data0()
                         : PhiTypes.Data0();
      TypeTree vd2 = isa<Constant>(bo->getOperand(1))
                         ? getAnalysis(bo->getOperand(1)).Data0()
                         : PhiTypes.Data0();
      bool Legal = true;
      vd1.binopIn(Legal, vd2, bo->getOpcode());
      if (!Legal) {
        std::string str;
        raw_string_ostream ss(str);
        if (!CustomErrorHandler) {
          llvm::errs() << *fntypeinfo.Function->getParent() << "\n";
          llvm::errs() << *fntypeinfo.Function << "\n";
          dump(ss);
        }
        ss << "Illegal updateBinop Analysis " << *bo << "\n";
        ss << "Illegal binopIn(consts): " << *bo << " lhs: " << vd1.str()
           << " rhs: " << vd2.str() << "\n";
        if (CustomErrorHandler) {
          CustomErrorHandler(str.c_str(), wrap(bo),
                             ErrorType::IllegalTypeAnalysis, (void *)this,
                             wrap(bo), nullptr);
        }
        EmitFailure("IllegalUpdateAnalysis", bo->getDebugLoc(), bo, ss.str());
        report_fatal_error("Performed illegal updateAnalysis");
      }
      PhiTypes &= vd1.Only(bo->getType()->isIntegerTy() ? -1 : 0, &phi);
    }

    if (direction & DOWN) {
      if (phi.getType()->isIntOrIntVectorTy() &&
          PhiTypes.Inner0() == BaseType::Anything) {
        if (mustRemainInteger(&phi)) {
          PhiTypes = TypeTree(BaseType::Integer).Only(-1, &phi);
        }
      }
      updateAnalysis(&phi, PhiTypes, &phi);
    }
  }
}

void TypeAnalyzer::visitTruncInst(TruncInst &I) {
  auto &DL = fntypeinfo.Function->getParent()->getDataLayout();
  size_t inSize = (DL.getTypeSizeInBits(I.getOperand(0)->getType()) + 7) / 8;
  size_t outSize = (DL.getTypeSizeInBits(I.getType()) + 7) / 8;
  if (direction & DOWN)
    if (outSize != 1)
      updateAnalysis(&I,
                     getAnalysis(I.getOperand(0))
                         .ShiftIndices(DL, /*off*/ 0, inSize, /*addOffset*/ 0)
                         .ShiftIndices(DL, /*off*/ 0, outSize, /*addOffset*/ 0),
                     &I);
  // Don't propagate up a trunc float -> i8
  if (direction & UP)
    if (outSize != 1 || inSize == 1)
      updateAnalysis(
          I.getOperand(0),
          getAnalysis(&I).ShiftIndices(DL, /*off*/ 0, outSize, /*addOffset*/ 0),
          &I);
}

void TypeAnalyzer::visitZExtInst(ZExtInst &I) {
  if (direction & DOWN) {
    TypeTree Result;
    if (cast<IntegerType>(I.getOperand(0)->getType()->getScalarType())
            ->getBitWidth() == 1) {
      Result = TypeTree(BaseType::Anything).Only(-1, &I);
    } else {
      Result = getAnalysis(I.getOperand(0));
    }

    if (I.getType()->isIntOrIntVectorTy() &&
        Result.Inner0() == BaseType::Anything) {
      if (mustRemainInteger(&I)) {
        Result = TypeTree(BaseType::Integer).Only(-1, &I);
      }
    }
    updateAnalysis(&I, Result, &I);
  }
  if (direction & UP) {
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
  }
}

void TypeAnalyzer::visitSExtInst(SExtInst &I) {
  // This is only legal on integer types [not pointers per sign]
  // nor floatings points. Likewise, there's no direction check
  // necessary since this is always valid.
  updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1, &I), &I);
  updateAnalysis(I.getOperand(0), TypeTree(BaseType::Integer).Only(-1, &I), &I);
}

void TypeAnalyzer::visitAddrSpaceCastInst(AddrSpaceCastInst &I) {
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitFPExtInst(FPExtInst &I) {
  // No direction check as always true
  updateAnalysis(
      &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1, &I),
      &I);
  updateAnalysis(
      I.getOperand(0),
      TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
          .Only(-1, &I),
      &I);
}

void TypeAnalyzer::visitFPTruncInst(FPTruncInst &I) {
  // No direction check as always true
  updateAnalysis(
      &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1, &I),
      &I);
  updateAnalysis(
      I.getOperand(0),
      TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
          .Only(-1, &I),
      &I);
}

void TypeAnalyzer::visitFPToUIInst(FPToUIInst &I) {
  // No direction check as always true
  updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1, &I), &I);
  updateAnalysis(
      I.getOperand(0),
      TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
          .Only(-1, &I),
      &I);
}

void TypeAnalyzer::visitFPToSIInst(FPToSIInst &I) {
  // No direction check as always true
  updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1, &I), &I);
  updateAnalysis(
      I.getOperand(0),
      TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
          .Only(-1, &I),
      &I);
}

void TypeAnalyzer::visitUIToFPInst(UIToFPInst &I) {
  // No direction check as always true
  updateAnalysis(I.getOperand(0), TypeTree(BaseType::Integer).Only(-1, &I), &I);
  updateAnalysis(
      &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1, &I),
      &I);
}

void TypeAnalyzer::visitSIToFPInst(SIToFPInst &I) {
  // No direction check as always true
  updateAnalysis(I.getOperand(0), TypeTree(BaseType::Integer).Only(-1, &I), &I);
  updateAnalysis(
      &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1, &I),
      &I);
}

void TypeAnalyzer::visitPtrToIntInst(PtrToIntInst &I) {
  // Note it is illegal to assume here that either is a pointer or an int
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitIntToPtrInst(IntToPtrInst &I) {
  // Note it is illegal to assume here that either is a pointer or an int
  if (direction & DOWN) {
    if (isa<ConstantInt>(I.getOperand(0))) {
      updateAnalysis(&I, TypeTree(BaseType::Anything).Only(-1, &I), &I);
    } else {
      updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
    }
  }
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitFreezeInst(FreezeInst &I) {
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitBitCastInst(BitCastInst &I) {
  if (direction & DOWN)
    updateAnalysis(&I, getAnalysis(I.getOperand(0)), &I);
  if (direction & UP)
    updateAnalysis(I.getOperand(0), getAnalysis(&I), &I);
}

void TypeAnalyzer::visitSelectInst(SelectInst &I) {
  if (direction & UP) {
    auto Data = getAnalysis(&I).PurgeAnything();
    if (EnzymeStrictAliasing || (I.getTrueValue() == I.getFalseValue())) {
      updateAnalysis(I.getTrueValue(), Data, &I);
      updateAnalysis(I.getFalseValue(), Data, &I);
    } else {
      if (EnzymePrintType) {
        llvm::errs() << " skipping update into ";
        I.getTrueValue()->print(llvm::errs(), *MST);
        llvm::errs() << " of " << Data.str() << " from ";
        I.print(llvm::errs(), *MST);
        llvm::errs() << "\n";
        llvm::errs() << " skipping update into ";
        I.getFalseValue()->print(llvm::errs(), *MST);
        llvm::errs() << " of " << Data.str() << " from ";
        I.print(llvm::errs(), *MST);
        llvm::errs() << "\n";
      }
    }
  }
  if (direction & DOWN) {
    // special case for min/max result is still that operand [even if something
    // is 0]
    if (auto cmpI = dyn_cast<CmpInst>(I.getCondition())) {
      // is relational equiv to not is equality
      if (!cmpI->isEquality())
        if ((cmpI->getOperand(0) == I.getTrueValue() &&
             cmpI->getOperand(1) == I.getFalseValue()) ||
            (cmpI->getOperand(1) == I.getTrueValue() &&
             cmpI->getOperand(0) == I.getFalseValue())) {
          auto vd = getAnalysis(I.getTrueValue()).Inner0();
          vd &= getAnalysis(I.getFalseValue()).Inner0();
          if (vd.isKnown()) {
            updateAnalysis(&I, TypeTree(vd).Only(-1, &I), &I);
            return;
          }
        }
    }
    // If getTrueValue and getFalseValue are the same type (per the and)
    // it is safe to assume the result is as well
    TypeTree vd = getAnalysis(I.getTrueValue()).PurgeAnything();
    vd &= getAnalysis(I.getFalseValue()).PurgeAnything();

    // A regular and operation, however is not sufficient. One of the operands
    // could be anything whereas the other is concrete, resulting in the
    // concrete type (e.g. select true, anything(0), integer(i64)) This is not
    // correct as the result of the select could always be anything (e.g. if it
    // is a pointer). As a result, explicitly or in any anything values
    // TODO this should be propagated elsewhere as well (specifically returns,
    // phi)
    TypeTree any = getAnalysis(I.getTrueValue()).JustAnything();
    any &= getAnalysis(I.getFalseValue()).JustAnything();
    vd |= any;
    updateAnalysis(&I, vd, &I);
  }
}

void TypeAnalyzer::visitExtractElementInst(ExtractElementInst &I) {
  updateAnalysis(I.getIndexOperand(), BaseType::Integer, &I);

  auto &dl = fntypeinfo.Function->getParent()->getDataLayout();
  VectorType *vecType = cast<VectorType>(I.getVectorOperand()->getType());

  size_t bitsize = dl.getTypeSizeInBits(vecType->getElementType());
  size_t size = (bitsize + 7) / 8;

  if (auto CI = dyn_cast<ConstantInt>(I.getIndexOperand())) {
    size_t off = (CI->getZExtValue() * bitsize) / 8;

    if (direction & DOWN)
      updateAnalysis(&I,
                     getAnalysis(I.getVectorOperand())
                         .ShiftIndices(dl, off, size, /*addOffset*/ 0),
                     &I);

    if (direction & UP)
      updateAnalysis(I.getVectorOperand(),
                     getAnalysis(&I).ShiftIndices(dl, 0, size, off), &I);

  } else {
    if (direction & DOWN) {
      TypeTree vecAnalysis = getAnalysis(I.getVectorOperand());
      // TODO merge of anythings (see selectinst)
      TypeTree res = vecAnalysis.Lookup(size, dl);
      updateAnalysis(&I, res.Only(-1, &I), &I);
    }
    if (direction & UP) {
      // propagated upward to unknown location, no analysis
      // can be updated
    }
  }
}

void TypeAnalyzer::visitInsertElementInst(InsertElementInst &I) {
  updateAnalysis(I.getOperand(2), TypeTree(BaseType::Integer).Only(-1, &I), &I);

  auto &dl = fntypeinfo.Function->getParent()->getDataLayout();
  VectorType *vecType = cast<VectorType>(I.getOperand(0)->getType());
  if (vecType->getElementType()->isIntegerTy(1)) {
    if (direction & UP) {
      updateAnalysis(I.getOperand(0), TypeTree(BaseType::Integer).Only(-1, &I),
                     &I);
      updateAnalysis(I.getOperand(1), TypeTree(BaseType::Integer).Only(-1, &I),
                     &I);
    }
    if (direction & DOWN) {
      updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1, &I), &I);
    }
    return;
  }
#if LLVM_VERSION_MAJOR >= 12
  assert(!vecType->getElementCount().isScalable());
  size_t numElems = vecType->getElementCount().getKnownMinValue();
#else
  size_t numElems = vecType->getNumElements();
#endif
  size_t size = (dl.getTypeSizeInBits(vecType->getElementType()) + 7) / 8;
  size_t vecSize = (dl.getTypeSizeInBits(vecType) + 7) / 8;

  if (auto CI = dyn_cast<ConstantInt>(I.getOperand(2))) {
    size_t off = CI->getZExtValue() * size;

    if (direction & UP)
      updateAnalysis(I.getOperand(0),
                     getAnalysis(&I).Clear(off, off + size, vecSize), &I);

    if (direction & UP)
      updateAnalysis(I.getOperand(1),
                     getAnalysis(&I).ShiftIndices(dl, off, size, 0), &I);

    if (direction & DOWN) {
      auto new_res =
          getAnalysis(I.getOperand(0)).Clear(off, off + size, vecSize);
      auto shifted =
          getAnalysis(I.getOperand(1)).ShiftIndices(dl, 0, size, off);
      new_res |= shifted;
      updateAnalysis(&I, new_res, &I);
    }
  } else {
    if (direction & DOWN) {
      auto new_res = getAnalysis(I.getOperand(0));
      auto inserted = getAnalysis(I.getOperand(1));
      // TODO merge of anythings (see selectinst)
      for (size_t i = 0; i < numElems; ++i)
        new_res &= inserted.ShiftIndices(dl, 0, size, size * i);
      updateAnalysis(&I, new_res, &I);
    }
  }
}

void TypeAnalyzer::visitShuffleVectorInst(ShuffleVectorInst &I) {
  // See selectinst type propagation rule for a description
  // of the ncessity and correctness of this rule.
  VectorType *resType = cast<VectorType>(I.getType());

  auto &dl = fntypeinfo.Function->getParent()->getDataLayout();

  const size_t lhs = 0;
  const size_t rhs = 1;

#if LLVM_VERSION_MAJOR >= 12
  assert(!cast<VectorType>(I.getOperand(lhs)->getType())
              ->getElementCount()
              .isScalable());
  size_t numFirst = cast<VectorType>(I.getOperand(lhs)->getType())
                        ->getElementCount()
                        .getKnownMinValue();
#else
  size_t numFirst =
      cast<VectorType>(I.getOperand(lhs)->getType())->getNumElements();
#endif
  size_t size = (dl.getTypeSizeInBits(resType->getElementType()) + 7) / 8;

  auto mask = I.getShuffleMask();

  TypeTree result; //  = getAnalysis(&I);
  for (size_t i = 0; i < mask.size(); ++i) {
    int newOff;
    {
      Value *vec[2] = {ConstantInt::get(Type::getInt64Ty(I.getContext()), 0),
                       ConstantInt::get(Type::getInt64Ty(I.getContext()), i)};
      auto ud = UndefValue::get(getUnqual(I.getOperand(0)->getType()));
      auto g2 = GetElementPtrInst::Create(I.getOperand(0)->getType(), ud, vec);
      APInt ai(dl.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
      g2->accumulateConstantOffset(dl, ai);
      // Using destructor rather than eraseFromParent
      //   as g2 has no parent
      delete g2;
      newOff = (int)ai.getLimitedValue();
      // there is a bug in LLVM, this is the correct offset
      if (cast<VectorType>(I.getOperand(lhs)->getType())
              ->getElementType()
              ->isIntegerTy(1)) {
        newOff = i / 8;
      }
    }
#if LLVM_VERSION_MAJOR > 16
    if (mask[i] == PoisonMaskElem)
#elif LLVM_VERSION_MAJOR >= 12
    if (mask[i] == UndefMaskElem)
#else
    if (mask[i] == -1)
#endif
    {
      if (direction & DOWN) {
        result |= TypeTree(BaseType::Anything)
                      .Only(-1, &I)
                      .ShiftIndices(dl, 0, size, newOff);
      }
    } else {
      if ((size_t)mask[i] < numFirst) {
        Value *vec[2] = {
            ConstantInt::get(Type::getInt64Ty(I.getContext()), 0),
            ConstantInt::get(Type::getInt64Ty(I.getContext()), mask[i])};
        auto ud = UndefValue::get(getUnqual(I.getOperand(0)->getType()));
        auto g2 =
            GetElementPtrInst::Create(I.getOperand(0)->getType(), ud, vec);
        APInt ai(dl.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
        g2->accumulateConstantOffset(dl, ai);
        // Using destructor rather than eraseFromParent
        //   as g2 has no parent
        int oldOff = (int)ai.getLimitedValue();
        // there is a bug in LLVM, this is the correct offset
        if (cast<VectorType>(I.getOperand(lhs)->getType())
                ->getElementType()
                ->isIntegerTy(1)) {
          oldOff = mask[i] / 8;
        }
        delete g2;
        if (direction & UP) {
          updateAnalysis(I.getOperand(lhs),
                         getAnalysis(&I).ShiftIndices(dl, newOff, size, oldOff),
                         &I);
        }
        if (direction & DOWN) {
          result |= getAnalysis(I.getOperand(lhs))
                        .ShiftIndices(dl, oldOff, size, newOff);
        }
      } else {
        Value *vec[2] = {ConstantInt::get(Type::getInt64Ty(I.getContext()), 0),
                         ConstantInt::get(Type::getInt64Ty(I.getContext()),
                                          mask[i] - numFirst)};
        auto ud = UndefValue::get(getUnqual(I.getOperand(0)->getType()));
        auto g2 =
            GetElementPtrInst::Create(I.getOperand(0)->getType(), ud, vec);
        APInt ai(dl.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
        g2->accumulateConstantOffset(dl, ai);
        // Using destructor rather than eraseFromParent
        //   as g2 has no parent
        int oldOff = (int)ai.getLimitedValue();
        // there is a bug in LLVM, this is the correct offset
        if (cast<VectorType>(I.getOperand(lhs)->getType())
                ->getElementType()
                ->isIntegerTy(1)) {
          oldOff = (mask[i] - numFirst) / 8;
        }
        delete g2;
        if (direction & UP) {
          updateAnalysis(I.getOperand(rhs),
                         getAnalysis(&I).ShiftIndices(dl, newOff, size, oldOff),
                         &I);
        }
        if (direction & DOWN) {
          result |= getAnalysis(I.getOperand(rhs))
                        .ShiftIndices(dl, oldOff, size, newOff);
        }
      }
    }
  }

  if (direction & DOWN) {
    updateAnalysis(&I, result, &I);
  }
}

void TypeAnalyzer::visitExtractValueInst(ExtractValueInst &I) {
  auto &dl = fntypeinfo.Function->getParent()->getDataLayout();
  SmallVector<Value *, 4> vec;
  vec.push_back(ConstantInt::get(Type::getInt64Ty(I.getContext()), 0));
  for (auto ind : I.indices()) {
    vec.push_back(ConstantInt::get(Type::getInt32Ty(I.getContext()), ind));
  }
  auto ud = UndefValue::get(getUnqual(I.getOperand(0)->getType()));
  auto g2 = GetElementPtrInst::Create(I.getOperand(0)->getType(), ud, vec);
  APInt ai(dl.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
  g2->accumulateConstantOffset(dl, ai);
  // Using destructor rather than eraseFromParent
  //   as g2 has no parent
  delete g2;

  int off = (int)ai.getLimitedValue();
  int size = dl.getTypeSizeInBits(I.getType()) / 8;

  if (direction & DOWN)
    updateAnalysis(&I,
                   getAnalysis(I.getOperand(0))
                       .ShiftIndices(dl, off, size, /*addOffset*/ 0),
                   &I);

  if (direction & UP)
    updateAnalysis(I.getOperand(0),
                   getAnalysis(&I).ShiftIndices(dl, 0, size, off), &I);
}

void TypeAnalyzer::visitInsertValueInst(InsertValueInst &I) {
  auto &dl = fntypeinfo.Function->getParent()->getDataLayout();
  SmallVector<Value *, 4> vec = {
      ConstantInt::get(Type::getInt64Ty(I.getContext()), 0)};
  for (auto ind : I.indices()) {
    vec.push_back(ConstantInt::get(Type::getInt32Ty(I.getContext()), ind));
  }
  auto ud = UndefValue::get(getUnqual(I.getOperand(0)->getType()));
  auto g2 = GetElementPtrInst::Create(I.getOperand(0)->getType(), ud, vec);
  APInt ai(dl.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
  g2->accumulateConstantOffset(dl, ai);
  delete g2;
  // Using destructor rather than eraseFromParent
  //   as g2 has no parent

  // Compute the offset at the next logical element [e.g. adding 1 to the last
  // index, carrying the value on overflow]
  for (ssize_t i = vec.size() - 1; i >= 0; i--) {
    auto CI = cast<ConstantInt>(vec[i]);
    auto val = CI->getZExtValue();
    if (i == 0) {
      vec[i] = ConstantInt::get(CI->getType(), val + 1);
      break;
    }
    auto subTy = GetElementPtrInst::getIndexedType(
        I.getOperand(0)->getType(), ArrayRef<Value *>(vec).slice(0, i));
    if (auto ST = dyn_cast<StructType>(subTy)) {
      if (val + 1 == ST->getNumElements()) {
        vec.erase(vec.begin() + i, vec.end());
        continue;
      }
      vec[i] = ConstantInt::get(CI->getType(), val + 1);
      break;
    } else {
      auto AT = cast<ArrayType>(subTy);
      if (val + 1 == AT->getNumElements()) {
        vec.erase(vec.begin() + i, vec.end());
        continue;
      }
      vec[i] = ConstantInt::get(CI->getType(), val + 1);
      break;
    }
  }
  g2 = GetElementPtrInst::Create(I.getOperand(0)->getType(), ud, vec);
  APInt aiend(dl.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
  g2->accumulateConstantOffset(dl, aiend);
  delete g2;

  int off = (int)ai.getLimitedValue();

  int agg_size = (dl.getTypeSizeInBits(I.getType()) + 7) / 8;
  int ins_size = (int)(aiend - ai).getLimitedValue();
  int ins2_size =
      (dl.getTypeSizeInBits(I.getInsertedValueOperand()->getType()) + 7) / 8;

  if (direction & UP)
    updateAnalysis(I.getAggregateOperand(),
                   getAnalysis(&I).Clear(off, off + ins_size, agg_size), &I);
  if (direction & UP)
    updateAnalysis(I.getInsertedValueOperand(),
                   getAnalysis(&I).ShiftIndices(dl, off, ins2_size, 0), &I);
  auto new_res =
      getAnalysis(I.getAggregateOperand()).Clear(off, off + ins_size, agg_size);
  auto shifted = getAnalysis(I.getInsertedValueOperand())
                     .ShiftIndices(dl, 0, ins_size, off);
  new_res |= shifted;
  if (direction & DOWN)
    updateAnalysis(&I, new_res, &I);
}

void TypeAnalyzer::dump(llvm::raw_ostream &ss) {
  ss << "<analysis>\n";
  // We don't care about correct MD node numbering here.
  ModuleSlotTracker MST(fntypeinfo.Function->getParent(),
                        /*ShouldInitializeAllMetadata*/ false);
  for (auto &pair : analysis) {
    if (auto F = dyn_cast<Function>(pair.first))
      ss << "@" << F->getName();
    else
      pair.first->print(ss, MST);
    ss << ": " << pair.second.str()
       << ", intvals: " << to_string(knownIntegralValues(pair.first)) << "\n";
  }
  ss << "</analysis>\n";
}

void TypeAnalyzer::visitAtomicRMWInst(llvm::AtomicRMWInst &I) {
  Value *Args[2] = {nullptr, I.getOperand(1)};
  TypeTree Ret = getAnalysis(&I);
  auto &DL = I.getParent()->getParent()->getParent()->getDataLayout();
  auto LoadSize = (DL.getTypeSizeInBits(I.getType()) + 7) / 8;
  TypeTree LHS = getAnalysis(I.getOperand(0)).Lookup(LoadSize, DL);
  TypeTree RHS = getAnalysis(I.getOperand(1));

  switch (I.getOperation()) {
  case AtomicRMWInst::Xchg: {
    auto tmp = LHS;
    LHS = RHS;
    RHS = tmp;
    bool Legal = true;
    LHS.checkedOrIn(Ret, /*PointerIntSame*/ false, Legal);
    if (!Legal) {
      dump();
      llvm::errs() << I << "\n";
      llvm::errs() << "Illegal orIn: " << LHS.str() << " right: " << Ret.str()
                   << "\n";
      llvm::errs() << *I.getOperand(0) << " "
                   << getAnalysis(I.getOperand(0)).str() << "\n";
      llvm::errs() << *I.getOperand(1) << " "
                   << getAnalysis(I.getOperand(1)).str() << "\n";
      assert(0 && "Performed illegal visitAtomicRMWInst::orIn");
      llvm_unreachable("Performed illegal visitAtomicRMWInst::orIn");
    }
    Ret = tmp;
    break;
  }
  case AtomicRMWInst::Add:
    visitBinaryOperation(DL, I.getType(), BinaryOperator::Add, Args, Ret, LHS,
                         RHS, &I);
    break;
  case AtomicRMWInst::Sub:
    visitBinaryOperation(DL, I.getType(), BinaryOperator::Sub, Args, Ret, LHS,
                         RHS, &I);
    break;
  case AtomicRMWInst::And:
    visitBinaryOperation(DL, I.getType(), BinaryOperator::And, Args, Ret, LHS,
                         RHS, &I);
    break;
  case AtomicRMWInst::Or:
    visitBinaryOperation(DL, I.getType(), BinaryOperator::Or, Args, Ret, LHS,
                         RHS, &I);
    break;
  case AtomicRMWInst::Xor:
    visitBinaryOperation(DL, I.getType(), BinaryOperator::Xor, Args, Ret, LHS,
                         RHS, &I);
    break;
  case AtomicRMWInst::FAdd:
    visitBinaryOperation(DL, I.getType(), BinaryOperator::FAdd, Args, Ret, LHS,
                         RHS, &I);
    break;
  case AtomicRMWInst::FSub:
    visitBinaryOperation(DL, I.getType(), BinaryOperator::FSub, Args, Ret, LHS,
                         RHS, &I);
    break;
  case AtomicRMWInst::Max:
  case AtomicRMWInst::Min:
  case AtomicRMWInst::UMax:
  case AtomicRMWInst::UMin:
  case AtomicRMWInst::Nand:
  default:
    break;
  }

  if (direction & UP) {
    TypeTree ptr = LHS.PurgeAnything()
                       .ShiftIndices(DL, /*start*/ 0, LoadSize, /*addOffset*/ 0)
                       .Only(-1, &I);
    ptr.insert({-1}, BaseType::Pointer);
    updateAnalysis(I.getOperand(0), ptr, &I);
    updateAnalysis(I.getOperand(1), RHS, &I);
  }

  if (direction & DOWN) {
    if (Ret[{-1}] == BaseType::Anything && LHS[{-1}] != BaseType::Anything)
      Ret = LHS;
    if (I.getType()->isIntOrIntVectorTy() && Ret[{-1}] == BaseType::Anything) {
      if (mustRemainInteger(&I)) {
        Ret = TypeTree(BaseType::Integer).Only(-1, &I);
      }
    }
    updateAnalysis(&I, Ret, &I);
  }
}

void TypeAnalyzer::visitBinaryOperation(const DataLayout &dl, llvm::Type *T,
                                        llvm::Instruction::BinaryOps Opcode,
                                        Value *Args[2], TypeTree &Ret,
                                        TypeTree &LHS, TypeTree &RHS,
                                        Instruction *origin) {
  if (Opcode == BinaryOperator::FAdd || Opcode == BinaryOperator::FSub ||
      Opcode == BinaryOperator::FMul || Opcode == BinaryOperator::FDiv ||
      Opcode == BinaryOperator::FRem) {
    auto ty = T->getScalarType();
    assert(ty->isFloatingPointTy());
    ConcreteType dt(ty);
    if (direction & UP) {
      bool LegalOr = true;
      auto Data = TypeTree(dt).Only(-1, nullptr);
      LHS.checkedOrIn(Data, /*PointerIntSame*/ false, LegalOr);
      if (CustomErrorHandler && !LegalOr) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Illegal updateAnalysis prev:" << LHS.str()
           << " new: " << Data.str() << "\n";
        ss << "val: " << *Args[0];
        ss << "origin: " << *origin;
        CustomErrorHandler(str.c_str(), wrap(Args[0]),
                           ErrorType::IllegalTypeAnalysis, (void *)this,
                           wrap(origin), nullptr);
      }
      RHS.checkedOrIn(Data, /*PointerIntSame*/ false, LegalOr);
      if (CustomErrorHandler && !LegalOr) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Illegal updateAnalysis prev:" << RHS.str()
           << " new: " << Data.str() << "\n";
        ss << "val: " << *Args[1];
        ss << "origin: " << *origin;
        CustomErrorHandler(str.c_str(), wrap(Args[1]),
                           ErrorType::IllegalTypeAnalysis, (void *)this,
                           wrap(origin), nullptr);
      }
    }
    if (direction & DOWN)
      Ret |= TypeTree(dt).Only(-1, nullptr);
  } else {
    auto size = (dl.getTypeSizeInBits(T) + 7) / 8;
    auto AnalysisLHS = LHS.Data0();
    auto AnalysisRHS = RHS.Data0();
    auto AnalysisRet = Ret.Data0();

    switch (Opcode) {
    case BinaryOperator::Sub:
      // ptr - ptr => int and int - int => int; thus int = a - b says only that
      // these are equal ptr - int => ptr and int - ptr => ptr; thus
      // howerver we do not want to propagate underlying ptr types since it's
      // legal to subtract unrelated pointer
      if (direction & UP) {
        if (AnalysisRet[{}] == BaseType::Integer) {
          LHS |= TypeTree(AnalysisRHS[{}]).PurgeAnything().Only(-1, nullptr);
          RHS |= TypeTree(AnalysisLHS[{}]).PurgeAnything().Only(-1, nullptr);
        }
        if (AnalysisRet[{}] == BaseType::Pointer) {
          if (AnalysisLHS[{}] == BaseType::Pointer) {
            RHS |= TypeTree(BaseType::Integer).Only(-1, nullptr);
          }
          if (AnalysisRHS[{}] == BaseType::Integer) {
            LHS |= TypeTree(BaseType::Pointer).Only(-1, nullptr);
          }
        }
      }
      break;

    case BinaryOperator::Add:
    case BinaryOperator::Mul:
      // if a + b or a * b == int, then a and b must be ints
      if (direction & UP) {
        if (AnalysisRet[{}] == BaseType::Integer) {
          LHS.orIn({-1}, BaseType::Integer);
          RHS.orIn({-1}, BaseType::Integer);
        }
      }
      break;

    case BinaryOperator::Xor:
      if (direction & UP)
        for (int i = 0; i < 2; ++i) {
          Type *FT = nullptr;
          if (!(FT = Ret.IsAllFloat(size, dl)))
            continue;
          // If ^ against 0b10000000000, the result is a float
          bool validXor = containsOnlyAtMostTopBit(Args[i], FT, dl);
          if (validXor) {
            bool Legal = true;
            ((i == 0) ? RHS : LHS)
                .checkedOrIn(TypeTree(FT).Only(-1, nullptr),
                             /*pointerintsame*/ false, Legal);

            if (!Legal) {
              std::string str;
              raw_string_ostream ss(str);
              if (!CustomErrorHandler) {
                llvm::errs() << *fntypeinfo.Function->getParent() << "\n";
                llvm::errs() << *fntypeinfo.Function << "\n";
                dump(ss);
              }
              ss << "Illegal updateBinop (xor up) Analysis " << *origin << "\n";
              ss << " (i=" << i << ") " << (i == 0 ? "RHS" : "LHS") << " "
                 << ((i == 0) ? RHS : LHS).str() << " FT from ret: " << *FT
                 << "\n";
              if (CustomErrorHandler) {
                CustomErrorHandler(str.c_str(), wrap(origin),
                                   ErrorType::IllegalTypeAnalysis, (void *)this,
                                   wrap(origin), nullptr);
              }
              EmitFailure("IllegalUpdateAnalysis", origin->getDebugLoc(),
                          origin, ss.str());
              report_fatal_error("Performed illegal updateAnalysis");
            }
          }
        }
      break;
    case BinaryOperator::Or:
      for (int i = 0; i < 2; ++i) {
        Type *FT = nullptr;
        if (!(FT = Ret.IsAllFloat(size, dl)))
          continue;
        // If | against a number only or'ing the exponent, the result is a float
        bool validXor = false;
        if (auto CIT = dyn_cast_or_null<ConstantInt>(Args[i])) {
          if (dl.getTypeSizeInBits(FT) != dl.getTypeSizeInBits(CIT->getType()))
            continue;
          auto CI = CIT->getValue();
#if LLVM_VERSION_MAJOR > 16
          if (CI.isZero())
#else
          if (CI.isNullValue())
#endif
          {
            validXor = true;
          } else if (
              !CI.isNegative() &&
              ((FT->isFloatTy()
#if LLVM_VERSION_MAJOR > 16
                && (CI & ~0b01111111100000000000000000000000ULL).isZero()
#else
                && (CI & ~0b01111111100000000000000000000000ULL).isNullValue()
#endif
                    ) ||
               (FT->isDoubleTy()
#if LLVM_VERSION_MAJOR > 16
                &&
                (CI &
                 ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                    .isZero()
#else
                &&
                (CI &
                 ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                    .isNullValue()
#endif
                    ))) {
            validXor = true;
          }
        } else if (auto CV = dyn_cast_or_null<ConstantVector>(Args[i])) {
          validXor = true;
          if (dl.getTypeSizeInBits(FT) !=
              dl.getTypeSizeInBits(CV->getOperand(i)->getType()))
            continue;
          for (size_t i = 0, end = CV->getNumOperands(); i < end; ++i) {
            auto CI = dyn_cast<ConstantInt>(CV->getOperand(i))->getValue();

#if LLVM_VERSION_MAJOR > 16
            if (CI.isZero())
#else
            if (CI.isNullValue())
#endif
            {
            } else if (
                !CI.isNegative() &&
                ((FT->isFloatTy()
#if LLVM_VERSION_MAJOR > 16
                  && (CI & ~0b01111111100000000000000000000000ULL).isZero()
#else
                  && (CI & ~0b01111111100000000000000000000000ULL).isNullValue()
#endif
                      ) ||
                 (FT->isDoubleTy()
#if LLVM_VERSION_MAJOR > 16
                  &&
                  (CI &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isZero()
#else
                  &&
                  (CI &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isNullValue()
#endif
                      ))) {
            } else
              validXor = false;
          }
        } else if (auto CV = dyn_cast_or_null<ConstantDataVector>(Args[i])) {
          validXor = true;
          if (dl.getTypeSizeInBits(FT) !=
              dl.getTypeSizeInBits(CV->getElementType()))
            continue;
          for (size_t i = 0, end = CV->getNumElements(); i < end; ++i) {
            auto CI = CV->getElementAsAPInt(i);
#if LLVM_VERSION_MAJOR > 16
            if (CI.isZero())
#else
            if (CI.isNullValue())
#endif
            {
            } else if (
                !CI.isNegative() &&
                ((FT->isFloatTy()
#if LLVM_VERSION_MAJOR > 16
                  && (CI & ~0b01111111100000000000000000000000ULL).isZero()
#else
                  && (CI & ~0b01111111100000000000000000000000ULL).isNullValue()
#endif
                      ) ||
                 (FT->isDoubleTy()
#if LLVM_VERSION_MAJOR > 16
                  &&
                  (CI &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isZero()
#else
                  &&
                  (CI &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isNullValue()
#endif
                      ))) {
            } else
              validXor = false;
          }
        }
        if (validXor) {
          ((i == 0) ? RHS : LHS) |= TypeTree(FT).Only(-1, nullptr);
        }
      }
      break;
    default:
      break;
    }

    if (direction & DOWN) {
      TypeTree Result = AnalysisLHS;
      bool Legal = true;
      Result.binopIn(Legal, AnalysisRHS, Opcode);
      if (!Legal) {
        std::string str;
        raw_string_ostream ss(str);
        if (!CustomErrorHandler) {
          llvm::errs() << *fntypeinfo.Function->getParent() << "\n";
          llvm::errs() << *fntypeinfo.Function << "\n";
          dump(ss);
        }
        ss << "Illegal updateBinop Analysis " << *origin << "\n";
        ss << "Illegal binopIn(down): " << Opcode << " lhs: " << Result.str()
           << " rhs: " << AnalysisRHS.str() << "\n";
        if (CustomErrorHandler) {
          CustomErrorHandler(str.c_str(), wrap(origin),
                             ErrorType::IllegalTypeAnalysis, (void *)this,
                             wrap(origin), nullptr);
        }
        EmitFailure("IllegalUpdateAnalysis", origin->getDebugLoc(), origin,
                    ss.str());
        report_fatal_error("Performed illegal updateAnalysis");
      }
      if (Opcode == BinaryOperator::And) {
        for (int i = 0; i < 2; ++i) {
          if (Args[i])
            for (auto andval :
                 fntypeinfo.knownIntegralValues(Args[i], DT, intseen, SE)) {
              if (andval <= 16 && andval >= 0) {
                Result = TypeTree(BaseType::Integer);
              } else if (andval < 0 && andval >= -64) {
                // If a small negative number, this just masks off the lower
                // bits in this case we can say that this is the same as the
                // other operand
                Result = (i == 0 ? AnalysisRHS : AnalysisLHS);
              }
            }
          // If we and a constant against an integer, the result remains an
          // integer
          if (Args[i] && isa<ConstantInt>(Args[i]) &&
              (i == 0 ? AnalysisRHS : AnalysisLHS).Inner0() ==
                  BaseType::Integer) {
            Result = TypeTree(BaseType::Integer);
          }
        }
      } else if (Opcode == BinaryOperator::Add ||
                 Opcode == BinaryOperator::Sub) {
        for (int i = 0; i < 2; ++i) {
          if (i == 1 || Opcode == BinaryOperator::Add)
            if (auto CI = dyn_cast_or_null<ConstantInt>(Args[i])) {
              if (CI->isNegative() || CI->isZero() ||
                  CI->getLimitedValue() <= 4096) {
                // If add/sub with zero, small, or negative number, the result
                // is equal to the type of the other operand (and we don't need
                // to assume this was an "anything")
                Result = (i == 0 ? AnalysisRHS : AnalysisLHS);
              }
            }
        }
      } else if (Opcode == BinaryOperator::Mul) {
        for (int i = 0; i < 2; ++i) {
          // If we mul a constant against an integer, the result remains an
          // integer
          if (Args[i] && isa<ConstantInt>(Args[i]) &&
              (i == 0 ? AnalysisRHS : AnalysisLHS)[{}] == BaseType::Integer) {
            Result = TypeTree(BaseType::Integer);
          }
        }
      } else if (Opcode == BinaryOperator::URem) {
        if (auto CI = dyn_cast_or_null<ConstantInt>(Args[1])) {
          // If rem with a small integer, the result is also a small integer
          if (CI->getLimitedValue() <= 4096) {
            Result = TypeTree(BaseType::Integer);
          }
        }
      } else if (Opcode == BinaryOperator::Xor) {
        for (int i = 0; i < 2; ++i) {
          Type *FT;
          if (!(FT = (i == 0 ? RHS : LHS).IsAllFloat(size, dl)))
            continue;
          // If ^ against 0b10000000000, the result is a float
          bool validXor = containsOnlyAtMostTopBit(Args[i], FT, dl);
          if (validXor) {
            Result = ConcreteType(FT);
          }
        }
      } else if (Opcode == BinaryOperator::Or) {
        for (int i = 0; i < 2; ++i) {
          Type *FT;
          if (!(FT = (i == 0 ? RHS : LHS).IsAllFloat(size, dl)))
            continue;
          // If & against 0b10000000000, the result is a float
          bool validXor = false;
          if (auto CIT = dyn_cast_or_null<ConstantInt>(Args[i])) {
            if (dl.getTypeSizeInBits(FT) !=
                dl.getTypeSizeInBits(CIT->getType()))
              continue;
            auto CI = CIT->getValue();
#if LLVM_VERSION_MAJOR > 16
            if (CI.isZero())
#else
            if (CI.isNullValue())
#endif
            {
              validXor = true;
            } else if (
                !CI.isNegative() &&
                ((FT->isFloatTy()
#if LLVM_VERSION_MAJOR > 16
                  && (CI & ~0b01111111100000000000000000000000ULL).isZero()
#else
                  && (CI & ~0b01111111100000000000000000000000ULL).isNullValue()
#endif
                      ) ||
                 (FT->isDoubleTy()
#if LLVM_VERSION_MAJOR > 16
                  &&
                  (CI &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isZero()
#else
                  &&
                  (CI &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isNullValue()
#endif
                      ))) {
              validXor = true;
            }
          } else if (auto CV = dyn_cast_or_null<ConstantVector>(Args[i])) {
            validXor = true;
            if (dl.getTypeSizeInBits(FT) !=
                dl.getTypeSizeInBits(CV->getOperand(i)->getType()))
              continue;
            for (size_t i = 0, end = CV->getNumOperands(); i < end; ++i) {
              auto CI = dyn_cast<ConstantInt>(CV->getOperand(i))->getValue();
#if LLVM_VERSION_MAJOR > 16
              if (CI.isZero())
#else
              if (CI.isNullValue())
#endif
              {
              } else if (
                  !CI.isNegative() &&
                  ((FT->isFloatTy()
#if LLVM_VERSION_MAJOR > 16
                    && (CI & ~0b01111111100000000000000000000000ULL).isZero()
#else
                    &&
                    (CI & ~0b01111111100000000000000000000000ULL).isNullValue()
#endif
                        ) ||
                   (FT->isDoubleTy()
#if LLVM_VERSION_MAJOR > 16
                    &&
                    (CI &
                     ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                        .isZero()
#else
                    &&
                    (CI &
                     ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                        .isNullValue()
#endif
                        ))) {
              } else
                validXor = false;
            }
          } else if (auto CV = dyn_cast_or_null<ConstantDataVector>(Args[i])) {
            validXor = true;
            if (dl.getTypeSizeInBits(FT) !=
                dl.getTypeSizeInBits(CV->getElementType()))
              continue;
            for (size_t i = 0, end = CV->getNumElements(); i < end; ++i) {
              auto CI = CV->getElementAsAPInt(i);
#if LLVM_VERSION_MAJOR > 16
              if (CI.isZero())
#else
              if (CI.isNullValue())
#endif
              {
              } else if (
                  !CI.isNegative() &&
                  ((FT->isFloatTy()
#if LLVM_VERSION_MAJOR > 16
                    && (CI & ~0b01111111100000000000000000000000ULL).isZero()
#else
                    &&
                    (CI & ~0b01111111100000000000000000000000ULL).isNullValue()
#endif
                        ) ||
                   (FT->isDoubleTy()
#if LLVM_VERSION_MAJOR > 16
                    &&
                    (CI &
                     ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                        .isZero()
#else
                    &&
                    (CI &
                     ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                        .isNullValue()
#endif
                        ))) {
              } else
                validXor = false;
            }
          }
          if (validXor) {
            Result = ConcreteType(FT);
          }
        }
      }

      Ret = Result.Only(-1, nullptr);
    }
  }
}
void TypeAnalyzer::visitBinaryOperator(BinaryOperator &I) {
  Value *Args[2] = {I.getOperand(0), I.getOperand(1)};
  TypeTree Ret = getAnalysis(&I);
  TypeTree LHS = getAnalysis(I.getOperand(0));
  TypeTree RHS = getAnalysis(I.getOperand(1));
  auto &DL = I.getParent()->getParent()->getParent()->getDataLayout();
  visitBinaryOperation(DL, I.getType(), I.getOpcode(), Args, Ret, LHS, RHS, &I);

  if (direction & UP) {
    updateAnalysis(I.getOperand(0), LHS, &I);
    updateAnalysis(I.getOperand(1), RHS, &I);
  }

  if (direction & DOWN) {
    if (I.getType()->isIntOrIntVectorTy() && Ret[{-1}] == BaseType::Anything) {
      if (mustRemainInteger(&I)) {
        Ret = TypeTree(BaseType::Integer).Only(-1, &I);
      }
    }
    updateAnalysis(&I, Ret, &I);
  }
}

void TypeAnalyzer::visitMemTransferInst(llvm::MemTransferInst &MTI) {
  visitMemTransferCommon(MTI);
}

void TypeAnalyzer::visitMemTransferCommon(llvm::CallBase &MTI) {
  if (MTI.getType()->isIntegerTy()) {
    updateAnalysis(&MTI, TypeTree(BaseType::Integer).Only(-1, &MTI), &MTI);
  }

  if (!(direction & UP))
    return;

  // If memcpy / memmove of pointer, we can propagate type information from src
  // to dst up to the length and vice versa
  size_t sz = 1;
  for (auto val :
       fntypeinfo.knownIntegralValues(MTI.getArgOperand(2), DT, intseen, SE)) {
    if (val >= 0) {
      sz = max(sz, (size_t)val);
    }
  }

  auto &dl = MTI.getParent()->getParent()->getParent()->getDataLayout();
  TypeTree res = getAnalysis(MTI.getArgOperand(0))
                     .PurgeAnything()
                     .Data0()
                     .ShiftIndices(dl, 0, sz, 0);
  TypeTree res2 = getAnalysis(MTI.getArgOperand(1))
                      .PurgeAnything()
                      .Data0()
                      .ShiftIndices(dl, 0, sz, 0);

  bool Legal = true;
  res.checkedOrIn(res2, /*PointerIntSame*/ false, Legal);
  if (!Legal) {
    std::string str;
    raw_string_ostream ss(str);
    if (!CustomErrorHandler) {
      llvm::errs() << *fntypeinfo.Function->getParent() << "\n";
      llvm::errs() << *fntypeinfo.Function << "\n";
      dump(ss);
    }
    ss << "Illegal updateMemTransfer Analysis " << MTI << "\n";
    ss << "Illegal orIn: " << res.str() << " right: " << res2.str() << "\n";
    ss << *MTI.getArgOperand(0) << " "
       << getAnalysis(MTI.getArgOperand(0)).str() << "\n";
    ss << *MTI.getArgOperand(1) << " "
       << getAnalysis(MTI.getArgOperand(1)).str() << "\n";

    if (CustomErrorHandler) {
      CustomErrorHandler(str.c_str(), wrap(&MTI),
                         ErrorType::IllegalTypeAnalysis, (void *)this,
                         wrap(&MTI), nullptr);
    }
    EmitFailure("IllegalUpdateAnalysis", MTI.getDebugLoc(), &MTI, ss.str());
    report_fatal_error("Performed illegal updateAnalysis");
  }
  res.insert({}, BaseType::Pointer);
  res = res.Only(-1, &MTI);
  updateAnalysis(MTI.getArgOperand(0), res, &MTI);
  updateAnalysis(MTI.getArgOperand(1), res, &MTI);
#if LLVM_VERSION_MAJOR >= 14
  for (unsigned i = 2; i < MTI.arg_size(); ++i)
#else
  for (unsigned i = 2; i < MTI.getNumArgOperands(); ++i)
#endif
  {
    updateAnalysis(MTI.getArgOperand(i),
                   TypeTree(BaseType::Integer).Only(-1, &MTI), &MTI);
  }
}

void TypeAnalyzer::visitIntrinsicInst(llvm::IntrinsicInst &I) {
  switch (I.getIntrinsicID()) {
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::nvvm_read_ptx_sreg_tid_x:
  case Intrinsic::nvvm_read_ptx_sreg_tid_y:
  case Intrinsic::nvvm_read_ptx_sreg_tid_z:
  case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
  case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
  case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
  case Intrinsic::nvvm_read_ptx_sreg_warpsize:
  case Intrinsic::amdgcn_workitem_id_x:
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::amdgcn_workitem_id_z:
    // No direction check as always valid
    updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1, &I), &I);
    return;

#if LLVM_VERSION_MAJOR < 22
  case Intrinsic::nvvm_barrier0_popc:
  case Intrinsic::nvvm_barrier0_and:
  case Intrinsic::nvvm_barrier0_or:
#else
  case Intrinsic::nvvm_barrier_cta_red_and_aligned_all:
  case Intrinsic::nvvm_barrier_cta_red_and_aligned_count:
  case Intrinsic::nvvm_barrier_cta_red_or_aligned_all:
  case Intrinsic::nvvm_barrier_cta_red_or_aligned_count:
  case Intrinsic::nvvm_barrier_cta_red_popc_aligned_all:
  case Intrinsic::nvvm_barrier_cta_red_popc_aligned_count:
#endif
    // No direction check as always valid
    updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1, &I), &I);
    updateAnalysis(I.getOperand(0), TypeTree(BaseType::Integer).Only(-1, &I),
                   &I);
    return;

  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_col:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_row:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_col:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_row:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_col:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_row:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_row_stride: {
    TypeTree TT;
    TT.insert({-1}, BaseType::Pointer);
    TT.insert({-1, 0}, Type::getFloatTy(I.getContext()));
    updateAnalysis(I.getOperand(0), TT, &I);
    for (int i = 1; i <= 9; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getFloatTy(I.getContext()))).Only(-1, &I),
          &I);
    return;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_row_stride: {
    TypeTree TT;
    TT.insert({-1}, BaseType::Pointer);
    TT.insert({-1, 0}, Type::getHalfTy(I.getContext()));
    updateAnalysis(I.getOperand(0), TT, &I);
    for (int i = 1; i <= 9; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
          &I);
    return;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_row_stride: {
    TypeTree TT;
    TT.insert({-1}, BaseType::Pointer);
    TT.insert({-1, 0}, Type::getFloatTy(I.getContext()));
    updateAnalysis(I.getOperand(0), TT, &I);
    updateAnalysis(
        &I,
        TypeTree(ConcreteType(Type::getFloatTy(I.getContext()))).Only(-1, &I),
        &I);
    return;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_row_stride: {
    TypeTree TT;
    TT.insert({-1}, BaseType::Pointer);
    TT.insert({-1, 0}, Type::getHalfTy(I.getContext()));
    updateAnalysis(I.getOperand(0), TT, &I);
    updateAnalysis(
        &I,
        TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
        &I);
    return;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_s8_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_a_u8_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_col_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_col:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_s8_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_row_stride:
  case Intrinsic::nvvm_wmma_m16n16k16_load_b_u8_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m16n16k16_load_c_s32_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_s8_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_a_u8_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_s8_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_row_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_b_u8_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m32n8k16_load_c_s32_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_s8_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_a_u8_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_s8_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_row_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_b_u8_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m8n32k16_load_c_s32_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k128_load_a_b1_row:
  case Intrinsic::nvvm_wmma_m8n8k128_load_a_b1_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k128_load_b_b1_col:
  case Intrinsic::nvvm_wmma_m8n8k128_load_b_b1_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k128_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m8n8k128_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k128_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m8n8k128_load_c_s32_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_a_s4_row:
  case Intrinsic::nvvm_wmma_m8n8k32_load_a_s4_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_a_u4_row_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_a_u4_row:
  case Intrinsic::nvvm_wmma_m8n8k32_load_b_s4_col:
  case Intrinsic::nvvm_wmma_m8n8k32_load_b_s4_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_b_u4_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_b_u4_col:
  case Intrinsic::nvvm_wmma_m8n8k32_load_c_s32_col:
  case Intrinsic::nvvm_wmma_m8n8k32_load_c_s32_col_stride:
  case Intrinsic::nvvm_wmma_m8n8k32_load_c_s32_row:
  case Intrinsic::nvvm_wmma_m8n8k32_load_c_s32_row_stride: {
    // TODO
    return;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_f16_f16:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_f16_f16:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_f16_f16:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f16_f16:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_f16_f16:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_f16_f16:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_f16_f16:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_f16_f16:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_f16_f16:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_f16_f16:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_f16_f16:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_f16_f16: {
    for (int i = 0; i < 16; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
          &I);
    for (int i = 16; i < 16 + 8; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
          &I);
    updateAnalysis(
        &I,
        TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
        &I);
    return;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_f16_f32:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_f16_f32:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_f16_f32:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f16_f32:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_f16_f32:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_f16_f32:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_f16_f32:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_f16_f32:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_f16_f32:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_f16_f32:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_f16_f32:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_f16_f32: {
    for (int i = 0; i < 16; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
          &I);
    for (int i = 16; i < 16 + 8; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getFloatTy(I.getContext()))).Only(-1, &I),
          &I);
    updateAnalysis(
        &I,
        TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
        &I);
    return;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_f32_f16:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_f32_f16:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_f32_f16:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f32_f16:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_f32_f16:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_f32_f16:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_f32_f16:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_f32_f16:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_f32_f16:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_f32_f16:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_f32_f16:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_f32_f16: {
    for (int i = 0; i < 16; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
          &I);
    for (int i = 16; i < 16 + 8; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
          &I);
    updateAnalysis(
        &I,
        TypeTree(ConcreteType(Type::getFloatTy(I.getContext()))).Only(-1, &I),
        &I);
    return;
  }

  case Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_f32_f32:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_f32_f32:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_f32_f32:
  case Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f32_f32:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_f32_f32:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_f32_f32:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_f32_f32:
  case Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_f32_f32:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_f32_f32:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_f32_f32:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_f32_f32:
  case Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_f32_f32: {
    for (int i = 0; i < 16; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getHalfTy(I.getContext()))).Only(-1, &I),
          &I);
    for (int i = 16; i < 16 + 8; i++)
      updateAnalysis(
          I.getOperand(i),
          TypeTree(ConcreteType(Type::getFloatTy(I.getContext()))).Only(-1, &I),
          &I);
    updateAnalysis(
        &I,
        TypeTree(ConcreteType(Type::getFloatTy(I.getContext()))).Only(-1, &I),
        &I);
    return;
  }

#if LLVM_VERSION_MAJOR < 20
  case Intrinsic::nvvm_ldg_global_i:
  case Intrinsic::nvvm_ldg_global_p:
  case Intrinsic::nvvm_ldg_global_f:
#endif
  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_p:
  case Intrinsic::nvvm_ldu_global_f: {
    auto &DL = I.getParent()->getParent()->getParent()->getDataLayout();
    auto LoadSize = (DL.getTypeSizeInBits(I.getType()) + 7) / 8;

    if (direction & UP) {
      TypeTree ptr(BaseType::Pointer);
      ptr |= getAnalysis(&I).PurgeAnything().ShiftIndices(
          DL, /*start*/ 0, LoadSize, /*addOffset*/ 0);
      updateAnalysis(I.getOperand(0), ptr.Only(-1, &I), &I);
    }
    if (direction & DOWN)
      updateAnalysis(&I, getAnalysis(I.getOperand(0)).Lookup(LoadSize, DL), &I);
    return;
  }

  case Intrinsic::log:
  case Intrinsic::log2:
  case Intrinsic::log10:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::sin:
  case Intrinsic::cos:
#if LLVM_VERSION_MAJOR >= 19
  case Intrinsic::sinh:
  case Intrinsic::cosh:
  case Intrinsic::tanh:
#endif
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
  case Intrinsic::sqrt:
#if LLVM_VERSION_MAJOR >= 21
  case Intrinsic::nvvm_fabs:
  case Intrinsic::nvvm_fabs_ftz:
#else
  case Intrinsic::nvvm_fabs_f:
  case Intrinsic::nvvm_fabs_d:
  case Intrinsic::nvvm_fabs_ftz_f:
#endif
  case Intrinsic::fabs:
    // No direction check as always valid
    updateAnalysis(
        &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1, &I),
        &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(0),
        TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
            .Only(-1, &I),
        &I);
    return;

  case Intrinsic::fmuladd:
  case Intrinsic::fma:
    // No direction check as always valid
    updateAnalysis(
        &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1, &I),
        &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(0),
        TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
            .Only(-1, &I),
        &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(1),
        TypeTree(ConcreteType(I.getOperand(1)->getType()->getScalarType()))
            .Only(-1, &I),
        &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(2),
        TypeTree(ConcreteType(I.getOperand(2)->getType()->getScalarType()))
            .Only(-1, &I),
        &I);
    return;

  case Intrinsic::powi:
    // No direction check as always valid
    updateAnalysis(
        &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1, &I),
        &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(0),
        TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
            .Only(-1, &I),
        &I);
    // No direction check as always valid
    updateAnalysis(I.getOperand(1), TypeTree(BaseType::Integer).Only(-1, &I),
                   &I);
    return;

#if LLVM_VERSION_MAJOR >= 12
  case Intrinsic::vector_reduce_fadd:
  case Intrinsic::vector_reduce_fmul:
#else
  case Intrinsic::experimental_vector_reduce_v2_fadd:
  case Intrinsic::experimental_vector_reduce_v2_fmul:
#endif
  case Intrinsic::copysign:
  case Intrinsic::maxnum:
  case Intrinsic::minnum:
#if LLVM_VERSION_MAJOR >= 15
  case Intrinsic::maximum:
  case Intrinsic::minimum:
#endif
  case Intrinsic::nvvm_fmax_f:
  case Intrinsic::nvvm_fmax_d:
  case Intrinsic::nvvm_fmax_ftz_f:
  case Intrinsic::nvvm_fmin_f:
  case Intrinsic::nvvm_fmin_d:
  case Intrinsic::nvvm_fmin_ftz_f:
  case Intrinsic::pow:
    // No direction check as always valid
    updateAnalysis(
        &I, TypeTree(ConcreteType(I.getType()->getScalarType())).Only(-1, &I),
        &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(0),
        TypeTree(ConcreteType(I.getOperand(0)->getType()->getScalarType()))
            .Only(-1, &I),
        &I);
    // No direction check as always valid
    updateAnalysis(
        I.getOperand(1),
        TypeTree(ConcreteType(I.getOperand(1)->getType()->getScalarType()))
            .Only(-1, &I),
        &I);
    return;
#if LLVM_VERSION_MAJOR >= 12
  case Intrinsic::smax:
  case Intrinsic::smin:
  case Intrinsic::umax:
  case Intrinsic::umin:
    if (direction & UP) {
      auto returnType = getAnalysis(&I)[{-1}];
      if (returnType == BaseType::Integer || returnType == BaseType::Pointer) {
        updateAnalysis(I.getOperand(0), TypeTree(returnType).Only(-1, &I), &I);
        updateAnalysis(I.getOperand(1), TypeTree(returnType).Only(-1, &I), &I);
      }
    }
    if (direction & DOWN) {
      auto opType0 = getAnalysis(I.getOperand(0))[{-1}];
      auto opType1 = getAnalysis(I.getOperand(1))[{-1}];
      if (opType0 == opType1 &&
          (opType0 == BaseType::Integer || opType0 == BaseType::Pointer)) {
        updateAnalysis(&I, TypeTree(opType0).Only(-1, &I), &I);
      } else if (opType0 == BaseType::Integer &&
                 opType1 == BaseType::Anything) {
        updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1, &I), &I);
      } else if (opType1 == BaseType::Integer &&
                 opType0 == BaseType::Anything) {
        updateAnalysis(&I, TypeTree(BaseType::Integer).Only(-1, &I), &I);
      }
    }
    return;
#endif
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow: {
    // val, bool
    auto analysis = getAnalysis(&I).Data0();

    BinaryOperator::BinaryOps opcode;
    // TODO update to use better rules in regular binop
    switch (I.getIntrinsicID()) {
    case Intrinsic::ssub_with_overflow:
    case Intrinsic::usub_with_overflow: {
      // TODO propagate this info
      // ptr - ptr => int and int - int => int; thus int = a - b says only that
      // these are equal ptr - int => ptr and int - ptr => ptr; thus
      analysis = ConcreteType(BaseType::Unknown);
      opcode = BinaryOperator::Sub;
      break;
    }

    case Intrinsic::smul_with_overflow:
    case Intrinsic::umul_with_overflow: {
      opcode = BinaryOperator::Mul;
      // if a + b or a * b == int, then a and b must be ints
      analysis = analysis.JustInt();
      break;
    }
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::uadd_with_overflow: {
      opcode = BinaryOperator::Add;
      // if a + b or a * b == int, then a and b must be ints
      analysis = analysis.JustInt();
      break;
    }
    default:
      llvm_unreachable("unknown binary operator");
    }

    // TODO update with newer binop protocol (see binop)
    if (direction & UP)
      updateAnalysis(I.getOperand(0), analysis.Only(-1, &I), &I);
    if (direction & UP)
      updateAnalysis(I.getOperand(1), analysis.Only(-1, &I), &I);

    TypeTree vd = getAnalysis(I.getOperand(0)).Data0();
    bool Legal = true;
    vd.binopIn(Legal, getAnalysis(I.getOperand(1)).Data0(), opcode);
    if (!Legal) {
      std::string str;
      raw_string_ostream ss(str);
      if (!CustomErrorHandler) {
        llvm::errs() << *fntypeinfo.Function->getParent() << "\n";
        llvm::errs() << *fntypeinfo.Function << "\n";
        dump(ss);
      }
      ss << "Illegal updateBinopIntr Analysis " << I << "\n";
      ss << "Illegal binopIn(intr): " << I << " lhs: " << vd.str()
         << " rhs: " << getAnalysis(I.getOperand(1)).str() << "\n";
      if (CustomErrorHandler) {
        CustomErrorHandler(str.c_str(), wrap(&I),
                           ErrorType::IllegalTypeAnalysis, (void *)this,
                           wrap(&I), nullptr);
      }
      EmitFailure("IllegalUpdateAnalysis", I.getDebugLoc(), &I, ss.str());
      report_fatal_error("Performed illegal updateAnalysis");
    }
    auto &dl = I.getParent()->getParent()->getParent()->getDataLayout();
    int sz = (dl.getTypeSizeInBits(I.getOperand(0)->getType()) + 7) / 8;
    TypeTree overall = vd.Only(-1, &I).ShiftIndices(dl, 0, sz, 0);

    int sz2 = (dl.getTypeSizeInBits(I.getType()) + 7) / 8;
    auto btree = TypeTree(BaseType::Integer)
                     .Only(-1, &I)
                     .ShiftIndices(dl, 0, sz2 - sz, sz);
    overall |= btree;

    if (direction & DOWN)
      updateAnalysis(&I, overall, &I);
    return;
  }
  default:
    return;
  }
}

/// This template class is defined to take the templated type T
/// update the analysis of the first argument (val) to be type T
/// As such, below we have several template specializations
/// to convert various c/c++ to TypeAnalysis types
template <typename T> struct TypeHandler {};

template <> struct TypeHandler<double> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TA.updateAnalysis(
        val,
        TypeTree(ConcreteType(Type::getDoubleTy(call.getContext())))
            .Only(-1, &call),
        &call);
  }
};

template <> struct TypeHandler<float> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TA.updateAnalysis(
        val,
        TypeTree(ConcreteType(Type::getFloatTy(call.getContext())))
            .Only(-1, &call),
        &call);
  }
};

template <> struct TypeHandler<long double> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TA.updateAnalysis(
        val,
        TypeTree(ConcreteType(Type::getX86_FP80Ty(call.getContext())))
            .Only(-1, &call),
        &call);
  }
};

#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
template <> struct TypeHandler<__float128> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TA.updateAnalysis(
        val,
        TypeTree(ConcreteType(Type::getFP128Ty(call.getContext())))
            .Only(-1, &call),
        &call);
  }
};
#endif

template <> struct TypeHandler<double *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(Type::getDoubleTy(call.getContext())).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<float *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(Type::getFloatTy(call.getContext())).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long double *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd =
        TypeTree(Type::getX86_FP80Ty(call.getContext())).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
template <> struct TypeHandler<__float128 *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(Type::getFP128Ty(call.getContext())).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};
#endif

template <> struct TypeHandler<void> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {}
};

template <> struct TypeHandler<void *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<int> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<int *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<unsigned int> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<unsigned int *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long int> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long int *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long unsigned int> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long unsigned int *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long long int> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long long int *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long long unsigned int> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <> struct TypeHandler<long long unsigned int *> {
  static void analyzeType(Value *val, CallBase &call, TypeAnalyzer &TA) {
    TypeTree vd = TypeTree(BaseType::Integer).Only(0, &call);
    vd |= TypeTree(BaseType::Pointer);
    TA.updateAnalysis(val, vd.Only(-1, &call), &call);
  }
};

template <typename... Arg0> struct FunctionArgumentIterator {
  static void analyzeFuncTypesHelper(unsigned idx, CallBase &call,
                                     TypeAnalyzer &TA) {}
};

template <typename Arg0, typename... Args>
struct FunctionArgumentIterator<Arg0, Args...> {
  static void analyzeFuncTypesHelper(unsigned idx, CallBase &call,
                                     TypeAnalyzer &TA) {
    TypeHandler<Arg0>::analyzeType(call.getOperand(idx), call, TA);
    FunctionArgumentIterator<Args...>::analyzeFuncTypesHelper(idx + 1, call,
                                                              TA);
  }
};

template <typename RT, typename... Args>
void analyzeFuncTypesNoFn(CallBase &call, TypeAnalyzer &TA) {
  TypeHandler<RT>::analyzeType(&call, call, TA);
  FunctionArgumentIterator<Args...>::analyzeFuncTypesHelper(0, call, TA);
}

template <typename RT, typename... Args>
void analyzeFuncTypes(RT (*fn)(Args...), CallBase &call, TypeAnalyzer &TA) {
  analyzeFuncTypesNoFn<RT, Args...>(call, TA);
}

void analyzeIntelSubscriptIntrinsic(IntrinsicInst &II, TypeAnalyzer &TA) {
  assert(isIntelSubscriptIntrinsic(II));
#if LLVM_VERSION_MAJOR >= 14
  assert(II.arg_size() == 5);
#else
  assert(II.getNumArgOperands() == 5);
#endif

  constexpr size_t idxArgsIndices[4] = {0, 1, 2, 4};
  constexpr size_t ptrArgIndex = 3;

  // Update analysis of index parameters

  if (TA.direction & TypeAnalyzer::UP) {
    for (auto i : idxArgsIndices) {
      auto idx = II.getOperand(i);
      TA.updateAnalysis(idx, TypeTree(BaseType::Integer).Only(-1, &II), &II);
    }
  }

  // Update analysis of ptr parameter

  auto &DL = TA.fntypeinfo.Function->getParent()->getDataLayout();
  auto pointerAnalysis = TA.getAnalysis(II.getOperand(ptrArgIndex));

  if (TA.direction & TypeAnalyzer::DOWN) {
    bool legal = true;
    auto keepMinus = pointerAnalysis.KeepMinusOne(legal);
    if (!legal) {
      if (CustomErrorHandler)
        CustomErrorHandler("Could not keep minus one", wrap(&II),
                           ErrorType::IllegalTypeAnalysis, &TA, nullptr,
                           nullptr);
      else {
        TA.dump();
        llvm::errs()
            << " could not perform minus one for llvm.intel.subscript'd: " << II
            << "\n";
      }
    }
    TA.updateAnalysis(&II, keepMinus, &II);
    TA.updateAnalysis(&II, TypeTree(pointerAnalysis.Inner0()).Only(-1, &II),
                      &II);
  }

  if (TA.direction & TypeAnalyzer::UP) {
    TA.updateAnalysis(II.getOperand(ptrArgIndex),
                      TypeTree(TA.getAnalysis(&II).Inner0()).Only(-1, &II),
                      &II);
  }

  SmallVector<std::set<int64_t>, 4> idnext;
  // The first operand is used to denote the axis of a multidimensional array,
  // but it is not used for address calculation, and so we skip it here.
  constexpr size_t offsetCalculationIndices[3] = {1, 2, 4};
  for (auto i : offsetCalculationIndices) {
    auto idx = II.getOperand(i);
    auto iset = TA.knownIntegralValues(idx);
    std::set<int64_t> vset;
    for (auto i : iset) {
      // Don't consider negative indices of llvm.intel.subscript
      if (i < 0)
        continue;
      vset.insert(i);
    }
    idnext.push_back(vset);
    if (idnext.back().size() == 0)
      return;
  }
  assert(idnext.size() != 0);

  TypeTree upTree;
  TypeTree downTree;

  TypeTree intrinsicData0;
  TypeTree pointerData0;
  if (TA.direction & TypeAnalyzer::UP)
    intrinsicData0 = TA.getAnalysis(&II).Data0();
  if (TA.direction & TypeAnalyzer::DOWN)
    pointerData0 = pointerAnalysis.Data0();

  bool firstLoop = true;

  for (auto vec : getSet<int64_t>(idnext, idnext.size() - 1)) {
    auto baseIndex = vec[0];
    auto stride = vec[1];
    auto index = vec[2];

    int offset = static_cast<int>(stride * (index - baseIndex));
    if (offset < 0) {
      continue; // The intrinsic doesn't handle negative offsets
    }

    if (TA.direction & TypeAnalyzer::DOWN) {
      auto shft = pointerData0.ShiftIndices(DL, /*init offset*/ offset,
                                            /*max size*/ -1, /*newoffset*/ 0);
      if (firstLoop)
        downTree = shft;
      else
        downTree &= shft;
    }

    if (TA.direction & TypeAnalyzer::UP) {
      auto shft =
          intrinsicData0.ShiftIndices(DL, /*init offset*/ 0, /*max size*/ -1,
                                      /*new offset*/ offset);
      if (firstLoop)
        upTree = shft;
      else
        upTree |= shft;
    }
    firstLoop = false;
  }
  if (TA.direction & TypeAnalyzer::DOWN)
    TA.updateAnalysis(&II, downTree.Only(-1, &II), &II);
  if (TA.direction & TypeAnalyzer::UP)
    TA.updateAnalysis(II.getOperand(ptrArgIndex), upTree.Only(-1, &II), &II);
}

void TypeAnalyzer::visitCallBase(CallBase &call) {
  assert(fntypeinfo.KnownValues.size() ==
         fntypeinfo.Function->getFunctionType()->getNumParams());

  if (auto iasm = dyn_cast<InlineAsm>(call.getCalledOperand())) {
    // NO direction check as always valid
    if (StringRef(iasm->getAsmString()).contains("cpuid")) {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
#if LLVM_VERSION_MAJOR >= 14
      for (auto &arg : call.args())
#else
      for (auto &arg : call.arg_operands())
#endif
      {
        updateAnalysis(arg, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      }
    }
  }

  if (call.hasFnAttr("enzyme_ta_norecur"))
    return;

  Function *ci = getFunctionFromCall(&call);

  if (ci) {
    if (ci->getAttributes().hasAttribute(AttributeList::FunctionIndex,
                                         "enzyme_ta_norecur"))
      return;

    StringRef funcName = getFuncNameFromCall(&call);

    auto blasMetaData = extractBLAS(funcName);
    if (blasMetaData) {
      BlasInfo blas = *blasMetaData;
#include "BlasTA.inc"
    }

    // clang-format off
    const char* NoTARecurStartsWith[] = {
      "std::__u::basic_ostream<wchar_t, std::__u::char_traits<wchar_t>>& std::__u::operator<<",
    };
    // clang-format on
    {
      std::string demangledName = llvm::demangle(funcName.str());
      // replace all '> >' with '>>'
      size_t start = 0;
      while ((start = demangledName.find("> >", start)) != std::string::npos) {
        demangledName.replace(start, 3, ">>");
      }
      for (auto Name : NoTARecurStartsWith)
        if (startsWith(demangledName, Name))
          return;
    }

    // Manual TT specification is non-interprocedural and already handled once
    // at the start.

    // When compiling Enzyme against standard LLVM, and not Intel's
    // modified version of LLVM, the intrinsic `llvm.intel.subscript` is
    // not fully understood by LLVM. One of the results of this is that the
    // visitor dispatches to visitCallBase, rather than visitIntrinsicInst, when
    // presented with the intrinsic - hence why we are handling it here.
    if (startsWith(funcName, "llvm.intel.subscript")) {
      assert(isa<IntrinsicInst>(call));
      analyzeIntelSubscriptIntrinsic(cast<IntrinsicInst>(call), *this);
      return;
    }

#define CONSIDER(fn)                                                           \
  if (funcName == #fn) {                                                       \
    analyzeFuncTypes(::fn, call, *this);                                       \
    return;                                                                    \
  }

#define CONSIDER2(fn, ...)                                                     \
  if (funcName == #fn) {                                                       \
    analyzeFuncTypesNoFn<__VA_ARGS__>(call, *this);                            \
    return;                                                                    \
  }

    auto customrule = interprocedural.CustomRules.find(funcName);
    if (customrule != interprocedural.CustomRules.end()) {
      auto returnAnalysis = getAnalysis(&call);
      SmallVector<TypeTree, 4> args;
      SmallVector<std::set<int64_t>, 4> knownValues;
#if LLVM_VERSION_MAJOR >= 14
      for (auto &arg : call.args())
#else
      for (auto &arg : call.arg_operands())
#endif
      {
        args.push_back(getAnalysis(arg));
        knownValues.push_back(
            fntypeinfo.knownIntegralValues((Value *)arg, DT, intseen, SE));
      }

      bool err = customrule->second(direction, returnAnalysis, args,
                                    knownValues, &call, this);
      if (err) {
        Invalid = true;
        return;
      }
      updateAnalysis(&call, returnAnalysis, &call);
      size_t argnum = 0;
#if LLVM_VERSION_MAJOR >= 14
      for (auto &arg : call.args())
#else
      for (auto &arg : call.arg_operands())
#endif
      {
        updateAnalysis(arg, args[argnum], &call);
        argnum++;
      }
      return;
    }

    // All these are always valid => no direction check
    // CONSIDER(malloc)
    // TODO consider handling other allocation functions integer inputs
    if (startsWith(funcName, "_ZN3std2io5stdio6_print") ||
        startsWith(funcName, "_ZN4core3fmt")) {
      return;
    }
    /// GEMM
    if (funcName == "dgemm_64" || funcName == "dgemm_64_" ||
        funcName == "dgemm" || funcName == "dgemm_") {
      TypeTree ptrint;
      ptrint.insert({-1}, BaseType::Pointer);
      ptrint.insert({-1, 0}, BaseType::Integer);
      // transa, transb, m, n, k, lda, ldb, ldc
      for (int i : {0, 1, 2, 3, 4, 7, 9, 12})
        updateAnalysis(call.getArgOperand(i), ptrint, &call);

      TypeTree ptrdbl;
      ptrdbl.insert({-1}, BaseType::Pointer);
      ptrdbl.insert({-1, 0}, Type::getDoubleTy(call.getContext()));

      // alpha, a, b, beta, c
      for (int i : {5, 6, 8, 10, 11})
        updateAnalysis(call.getArgOperand(i), ptrdbl, &call);
      return;
    }

    if (funcName == "__kmpc_fork_call") {
      Function *fn = dyn_cast<Function>(call.getArgOperand(2));

      if (auto castinst = dyn_cast<ConstantExpr>(call.getArgOperand(2)))
        if (castinst->isCast())
          fn = dyn_cast<Function>(castinst->getOperand(0));

      if (fn) {
#if LLVM_VERSION_MAJOR >= 14
        if (call.arg_size() - 3 != fn->getFunctionType()->getNumParams() - 2)
          return;
#else
        if (call.getNumArgOperands() - 3 !=
            fn->getFunctionType()->getNumParams() - 2)
          return;
#endif

        if (direction & UP) {
          FnTypeInfo typeInfo(fn);

          TypeTree IntPtr;
          IntPtr.insert({-1, -1}, BaseType::Integer);
          IntPtr.insert({-1}, BaseType::Pointer);

          int argnum = 0;
          for (auto &arg : fn->args()) {
            if (argnum <= 1) {
              typeInfo.Arguments.insert(
                  std::pair<Argument *, TypeTree>(&arg, IntPtr));
              typeInfo.KnownValues.insert(
                  std::pair<Argument *, std::set<int64_t>>(&arg, {0}));
            } else {
              typeInfo.Arguments.insert(std::pair<Argument *, TypeTree>(
                  &arg, getAnalysis(call.getArgOperand(argnum - 2 + 3))));
              std::set<int64_t> bounded;
              for (auto v : fntypeinfo.knownIntegralValues(
                       call.getArgOperand(argnum - 2 + 3), DT, intseen, SE)) {
                if (abs(v) > MaxIntOffset)
                  continue;
                bounded.insert(v);
              }
              typeInfo.KnownValues.insert(
                  std::pair<Argument *, std::set<int64_t>>(&arg, bounded));
            }

            ++argnum;
          }

          if (EnzymePrintType) {
            llvm::errs() << " starting omp IPO of ";
            call.print(llvm::errs(), *MST);
            llvm::errs() << "\n";
          }

          auto a = fn->arg_begin();
          ++a;
          ++a;
          TypeResults STR = interprocedural.analyzeFunction(typeInfo);
#if LLVM_VERSION_MAJOR >= 14
          for (unsigned i = 3; i < call.arg_size(); ++i)
#else
          for (unsigned i = 3; i < call.getNumArgOperands(); ++i)
#endif
          {
            auto dt = STR.query(a);
            updateAnalysis(call.getArgOperand(i), dt, &call);
            ++a;
          }
        }
      }
      return;
    }
    if (funcName == "__kmpc_for_static_init_4" ||
        funcName == "__kmpc_for_static_init_4u" ||
        funcName == "__kmpc_for_static_init_8" ||
        funcName == "__kmpc_for_static_init_8u") {
      TypeTree ptrint;
      ptrint.insert({-1}, BaseType::Pointer);
      size_t numBytes = 4;
      if (funcName == "__kmpc_for_static_init_8" ||
          funcName == "__kmpc_for_static_init_8u")
        numBytes = 8;
      for (size_t i = 0; i < numBytes; i++)
        ptrint.insert({-1, (int)i}, BaseType::Integer);
      updateAnalysis(call.getArgOperand(3), ptrint, &call);
      updateAnalysis(call.getArgOperand(4), ptrint, &call);
      updateAnalysis(call.getArgOperand(5), ptrint, &call);
      updateAnalysis(call.getArgOperand(6), ptrint, &call);
      updateAnalysis(call.getArgOperand(7),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getArgOperand(8),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "omp_get_max_threads" || funcName == "omp_get_thread_num" ||
        funcName == "omp_get_num_threads" ||
        funcName == "__kmpc_global_thread_num") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "_ZNSt6localeC1Ev") {
      TypeTree ptrint;
      ptrint.insert({-1}, BaseType::Pointer);
      ptrint.insert({-1, 0}, BaseType::Integer);
      updateAnalysis(call.getOperand(0), ptrint, &call);
      return;
    }

    if (startsWith(funcName, "_ZNKSt3__14hash")) {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }

    if (startsWith(funcName, "_ZNKSt3__112basic_string") ||
        startsWith(funcName, "_ZNSt3__112basic_string") ||
        startsWith(funcName, "_ZNSt3__112__hash_table") ||
        startsWith(funcName, "_ZNKSt3__115basic_stringbuf")) {
      return;
    }

    if (funcName == "__dynamic_cast" ||
        funcName == "_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base") {
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "memcmp") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }

    /// CUDA
    if (funcName == "cuDeviceGet") {
      // cuResult
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "cuDeviceGetName") {
      // cuResult
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "cudaRuntimeGetVersion" ||
        funcName == "cuDriverGetVersion" || funcName == "cuDeviceGetCount") {
      // cuResult
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      TypeTree ptrint;
      ptrint.insert({-1}, BaseType::Pointer);
      ptrint.insert({-1, 0}, BaseType::Integer);
      updateAnalysis(call.getOperand(0), ptrint, &call);
      return;
    }
    if (funcName == "cuMemGetInfo_v2") {
      // cuResult
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      TypeTree ptrint;
      ptrint.insert({-1}, BaseType::Pointer);
      ptrint.insert({-1, 0}, BaseType::Integer);
      updateAnalysis(call.getOperand(0), ptrint, &call);
      updateAnalysis(call.getOperand(1), ptrint, &call);
      return;
    }
    if (funcName == "cuDevicePrimaryCtxRetain" ||
        funcName == "cuCtxGetCurrent") {
      // cuResult
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "cuStreamQuery") {
      // cuResult
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "cuMemAllocAsync" || funcName == "cuMemAlloc" ||
        funcName == "cuMemAlloc_v2" || funcName == "cudaMalloc" ||
        funcName == "cudaMallocAsync" || funcName == "cudaMallocHost" ||
        funcName == "cudaMallocFromPoolAsync") {
      TypeTree ptrptr;
      ptrptr.insert({-1}, BaseType::Pointer);
      ptrptr.insert({-1, 0}, BaseType::Pointer);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0), ptrptr, &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "jl_hrtime" || funcName == "ijl_hrtime") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "jl_get_task_tid" || funcName == "ijl_get_task_tid") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "jl_get_binding_or_error" ||
        funcName == "ijl_get_binding_or_error") {
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "julia.gc_loaded") {
      if (direction & UP)
        updateAnalysis(call.getArgOperand(1), getAnalysis(&call), &call);
      if (direction & DOWN)
        updateAnalysis(&call, getAnalysis(call.getArgOperand(1)), &call);
      return;
    }
    if (funcName == "julia.pointer_from_objref") {
      if (direction & UP)
        updateAnalysis(call.getArgOperand(0), getAnalysis(&call), &call);
      if (direction & DOWN)
        updateAnalysis(&call, getAnalysis(call.getArgOperand(0)), &call);
      return;
    }
    if (funcName == "_ZNSt6chrono3_V212steady_clock3nowEv") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }

    /// MPI
    if (startsWith(funcName, "PMPI_"))
      funcName = funcName.substr(1);
    if (funcName == "MPI_Init") {
      TypeTree ptrint;
      ptrint.insert({-1}, BaseType::Pointer);
      ptrint.insert({-1, 0}, BaseType::Integer);
      updateAnalysis(call.getOperand(0), ptrint, &call);
      TypeTree ptrptrptr;
      ptrptrptr.insert({-1}, BaseType::Pointer);
      ptrptrptr.insert({-1, -1}, BaseType::Pointer);
      ptrptrptr.insert({-1, -1, 0}, BaseType::Pointer);
      updateAnalysis(call.getOperand(1), ptrptrptr, &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Comm_size" || funcName == "MPI_Comm_rank" ||
        funcName == "MPI_Get_processor_name") {
      TypeTree ptrint;
      ptrint.insert({-1}, BaseType::Pointer);
      ptrint.insert({-1, 0}, BaseType::Integer);
      updateAnalysis(call.getOperand(1), ptrint, &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Barrier" || funcName == "MPI_Finalize") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Send" || funcName == "MPI_Ssend" ||
        funcName == "MPI_Bsend" || funcName == "MPI_Recv" ||
        funcName == "MPI_Brecv" || funcName == "PMPI_Send" ||
        funcName == "PMPI_Ssend" || funcName == "PMPI_Bsend" ||
        funcName == "PMPI_Recv" || funcName == "PMPI_Brecv") {
      TypeTree buf = TypeTree(BaseType::Pointer);

      if (Constant *C = dyn_cast<Constant>(call.getOperand(2))) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_double") {
            buf.insert({0}, Type::getDoubleTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_float") {
            buf.insert({0}, Type::getFloatTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_cxx_bool") {
            buf.insert({0}, BaseType::Integer);
          }
        } else if (auto CI = dyn_cast<ConstantInt>(C)) {
          // MPICH
          if (CI->getValue() == 1275070475) {
            buf.insert({0}, Type::getDoubleTy(C->getContext()));
          } else if (CI->getValue() == 1275069450) {
            buf.insert({0}, Type::getFloatTy(C->getContext()));
          }
        }
      }
      updateAnalysis(call.getOperand(0), buf.Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(4),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Isend" || funcName == "MPI_Irecv" || funcName == "MPI_Send_init" || funcName == "MPI_Recv_init" ||
        funcName == "PMPI_Isend" || funcName == "PMPI_Irecv" || funcName == "PMPI_Send_init" || funcName == "PMPI_Recv_init") {
      TypeTree buf = TypeTree(BaseType::Pointer);

      if (Constant *C = dyn_cast<Constant>(call.getOperand(2))) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_double") {
            buf.insert({0}, Type::getDoubleTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_float") {
            buf.insert({0}, Type::getFloatTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_cxx_bool") {
            buf.insert({0}, BaseType::Integer);
          }
        } else if (auto CI = dyn_cast<ConstantInt>(C)) {
          // MPICH
          if (CI->getValue() == 1275070475) {
            buf.insert({0}, Type::getDoubleTy(C->getContext()));
          } else if (CI->getValue() == 1275069450) {
            buf.insert({0}, Type::getFloatTy(C->getContext()));
          }
        }
      }
      updateAnalysis(call.getOperand(0), buf.Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(4),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(6),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Start" || funcName == "PMPI_Start") {
      // TODO
      return; 
    }
    if (funcName == "MPI_Wait") {
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Waitany") {
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Waitall") {
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Bcast") {
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Reduce" || funcName == "PMPI_Reduce") {
      TypeTree buf = TypeTree(BaseType::Pointer);

      if (Constant *C = dyn_cast<Constant>(call.getOperand(3))) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_double") {
            buf.insert({0}, Type::getDoubleTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_float") {
            buf.insert({0}, Type::getFloatTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_cxx_bool") {
            buf.insert({0}, BaseType::Integer);
          }
        } else if (auto CI = dyn_cast<ConstantInt>(C)) {
          // MPICH
          if (CI->getValue() == 1275070475) {
            buf.insert({0}, Type::getDoubleTy(C->getContext()));
          } else if (CI->getValue() == 1275069450) {
            buf.insert({0}, Type::getFloatTy(C->getContext()));
          }
        }
      }
      // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
      // MPI_Datatype datatype,
      //         MPI_Op op, int root, MPI_Comm comm)
      // sendbuf
      updateAnalysis(call.getOperand(0), buf.Only(-1, &call), &call);
      // recvbuf
      updateAnalysis(call.getOperand(1), buf.Only(-1, &call), &call);
      // count
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      // datatype
      // op
      // comm
      // result
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Allreduce" || funcName == "PMPI_Allreduce") {
      TypeTree buf = TypeTree(BaseType::Pointer);

      if (Constant *C = dyn_cast<Constant>(call.getOperand(3))) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_double") {
            buf.insert({0}, Type::getDoubleTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_float") {
            buf.insert({0}, Type::getFloatTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_cxx_bool") {
            buf.insert({0}, BaseType::Integer);
          }
        } else if (auto CI = dyn_cast<ConstantInt>(C)) {
          // MPICH
          if (CI->getValue() == 1275070475) {
            buf.insert({0}, Type::getDoubleTy(C->getContext()));
          } else if (CI->getValue() == 1275069450) {
            buf.insert({0}, Type::getFloatTy(C->getContext()));
          }
        }
      }
      // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
      //             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
      // sendbuf
      updateAnalysis(call.getOperand(0), buf.Only(-1, &call), &call);
      // recvbuf
      updateAnalysis(call.getOperand(1), buf.Only(-1, &call), &call);
      // count
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      // datatype
      // op
      // comm
      // result
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Sendrecv_replace") {
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(4),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(5),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(6),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(8),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Sendrecv") {
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(4),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(5),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(6),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(7),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(8),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(9),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(11),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Gather" || funcName == "MPI_Scatter") {
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(4),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(6),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "MPI_Allgather") {
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(4),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    /// END MPI

    // Prob Prog
    if (ci->hasFnAttribute("enzyme_notypeanalysis")) {
      return;
    }

    if (funcName == "memcpy" || funcName == "memmove") {
      // TODO have this call common mem transfer to copy data
      visitMemTransferCommon(call);
      return;
    }
    if (funcName == "posix_memalign") {
      TypeTree ptrptr;
      ptrptr.insert({-1}, BaseType::Pointer);
      ptrptr.insert({-1, 0}, BaseType::Pointer);
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0), ptrptr, &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "calloc") {
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (auto opidx = getAllocationIndexFromCall(&call)) {
      auto ptr = TypeTree(BaseType::Pointer);
      unsigned index = (size_t)*opidx;
      if (auto CI = dyn_cast<ConstantInt>(call.getOperand(index))) {
        auto &DL = call.getParent()->getParent()->getParent()->getDataLayout();
        auto LoadSize = CI->getZExtValue();
        // Only propagate mappings in range that aren't "Anything" into the
        // pointer
        ptr |= getAnalysis(&call).Lookup(LoadSize, DL);
      }
      updateAnalysis(&call, ptr.Only(-1, &call), &call);
      updateAnalysis(call.getOperand(index),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "malloc") {
      auto ptr = TypeTree(BaseType::Pointer);
      if (auto CI = dyn_cast<ConstantInt>(call.getOperand(0))) {
        auto &DL = call.getParent()->getParent()->getParent()->getDataLayout();
        auto LoadSize = CI->getZExtValue();
        // Only propagate mappings in range that aren't "Anything" into the
        // pointer
        ptr |= getAnalysis(&call).Lookup(LoadSize, DL);
      }
      updateAnalysis(&call, ptr.Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "__size_returning_new_experiment") {
      auto ptr = TypeTree(BaseType::Pointer);
      auto &DL = call.getParent()->getParent()->getParent()->getDataLayout();
      if (auto CI = dyn_cast<ConstantInt>(call.getOperand(0))) {
        auto LoadSize = CI->getZExtValue();
        // Only propagate mappings in range that aren't "Anything" into the
        // pointer
        ptr |= getAnalysis(&call).Lookup(LoadSize, DL);
      }
      ptr = ptr.Only(0, &call);
      ptr |= TypeTree(BaseType::Integer).Only(DL.getPointerSize(), &call);
      updateAnalysis(&call, ptr.Only(0, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "julia.gc_alloc_obj" || funcName == "jl_gc_alloc_typed" ||
        funcName == "ijl_gc_alloc_typed") {
      auto ptr = TypeTree(BaseType::Pointer);
      if (auto CI = dyn_cast<ConstantInt>(call.getOperand(1))) {
        auto &DL = call.getParent()->getParent()->getParent()->getDataLayout();
        auto LoadSize = CI->getZExtValue();
        // Only propagate mappings in range that aren't "Anything" into the
        // pointer
        ptr |= getAnalysis(&call).Lookup(LoadSize, DL);
      }
      updateAnalysis(&call, ptr.Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "julia.except_enter" || funcName == "ijl_excstack_state" ||
        funcName == "jl_excstack_state") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "jl_array_copy" || funcName == "ijl_array_copy" ||
        funcName == "jl_inactive_inout" ||
        funcName == "jl_genericmemory_copy_slice" ||
        funcName == "ijl_genericmemory_copy_slice") {
      if (direction & DOWN)
        updateAnalysis(&call, getAnalysis(call.getOperand(0)), &call);
      if (direction & UP)
        updateAnalysis(call.getOperand(0), getAnalysis(&call), &call);
      return;
    }

    if (isAllocationFunction(funcName, TLI)) {
      size_t Idx = 0;
      for (auto &Arg : ci->args()) {
        if (Arg.getType()->isIntegerTy()) {
          updateAnalysis(call.getOperand(Idx),
                         TypeTree(BaseType::Integer).Only(-1, &call), &call);
        }
        Idx++;
      }
      assert(ci->getReturnType()->isPointerTy());
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "malloc_usable_size" || funcName == "malloc_size" ||
        funcName == "_msize") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "realloc") {
      size_t sz = 1;
      for (auto val : fntypeinfo.knownIntegralValues(call.getArgOperand(1), DT,
                                                     intseen, SE)) {
        if (val >= 0) {
          sz = max(sz, (size_t)val);
        }
      }

      auto &dl = call.getParent()->getParent()->getParent()->getDataLayout();
      TypeTree res = getAnalysis(call.getArgOperand(0))
                         .PurgeAnything()
                         .Data0()
                         .ShiftIndices(dl, 0, sz, 0);
      TypeTree res2 =
          getAnalysis(&call).PurgeAnything().Data0().ShiftIndices(dl, 0, sz, 0);

      res.orIn(res2, /*PointerIntSame*/ false);
      res.insert({}, BaseType::Pointer);
      res = res.Only(-1, &call);
      if (direction & DOWN) {
        updateAnalysis(&call, res, &call);
      }
      if (direction & UP) {
        updateAnalysis(call.getOperand(0), res, &call);
      }
      return;
    }
    if (funcName == "sigaction") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "mmap") {
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(3),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(4),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(5),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "munmap") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "pthread_mutex_lock" ||
        funcName == "pthread_mutex_trylock" ||
        funcName == "pthread_rwlock_rdlock" ||
        funcName == "pthread_rwlock_unlock" ||
        funcName == "pthread_attr_init" || funcName == "pthread_attr_destroy" ||
        funcName == "pthread_rwlock_unlock" ||
        funcName == "pthread_mutex_unlock") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (isDeallocationFunction(funcName, TLI)) {
      size_t Idx = 0;
      for (auto &Arg : ci->args()) {
        if (Arg.getType()->isIntegerTy()) {
          updateAnalysis(call.getOperand(Idx),
                         TypeTree(BaseType::Integer).Only(-1, &call), &call);
        }
        if (Arg.getType()->isPointerTy()) {
          updateAnalysis(call.getOperand(Idx),
                         TypeTree(BaseType::Pointer).Only(-1, &call), &call);
        }
        Idx++;
      }
      if (!ci->getReturnType()->isVoidTy()) {
        updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call),
                       &call);
        return;
      }
      assert(ci->getReturnType()->isVoidTy());
      return;
    }
    if (funcName == "memchr" || funcName == "memrchr") {
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "strlen") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "strcmp") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "bcmp") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "getcwd") {
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "sysconf") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "dladdr") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "__errno_location") {
      TypeTree ptrint;
      ptrint.insert({-1, -1}, BaseType::Integer);
      ptrint.insert({-1}, BaseType::Pointer);
      updateAnalysis(&call, ptrint, &call);
      return;
    }
    if (funcName == "getenv") {
      TypeTree ptrint;
      ptrint.insert({-1, -1}, BaseType::Integer);
      ptrint.insert({-1}, BaseType::Pointer);
      updateAnalysis(&call, ptrint, &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "getcwd") {
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "mprotect") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "memcmp") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "signal") {
      updateAnalysis(&call, TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "write" || funcName == "read" || funcName == "writev" ||
        funcName == "readv") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      // FD type not going to be defined here
      // updateAnalysis(call.getOperand(0),
      // TypeTree(BaseType::Pointer).Only(-1),
      //               &call);
      updateAnalysis(call.getOperand(1),
                     TypeTree(BaseType::Pointer).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(2),
                     TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }
    if (funcName == "gsl_sf_legendre_array_e") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
      return;
    }

    // CONSIDER(__lgamma_r_finite)

    CONSIDER2(frexp, double, double, int *)
    CONSIDER(frexpf)
    CONSIDER(frexpl)
    CONSIDER2(ldexp, double, double, int)
    CONSIDER2(modf, double, double, double *)
    CONSIDER(modff)
    CONSIDER(modfl)

    CONSIDER2(remquo, double, double, double, int *)
    CONSIDER(remquof)
    CONSIDER(remquol)

    if (isMemFreeLibMFunction(funcName)) {
#if LLVM_VERSION_MAJOR >= 14
      for (size_t i = 0; i < call.arg_size(); ++i)
#else
      for (size_t i = 0; i < call.getNumArgOperands(); ++i)
#endif
      {
        Type *T = call.getArgOperand(i)->getType();
        if (T->isFloatingPointTy()) {
          updateAnalysis(
              call.getArgOperand(i),
              TypeTree(ConcreteType(
                           call.getArgOperand(i)->getType()->getScalarType()))
                  .Only(-1, &call),
              &call);
        } else if (T->isIntegerTy()) {
          updateAnalysis(call.getArgOperand(i),
                         TypeTree(BaseType::Integer).Only(-1, &call), &call);
        } else if (auto ST = dyn_cast<StructType>(T)) {
          assert(ST->getNumElements() >= 1);
          for (size_t i = 1; i < ST->getNumElements(); ++i) {
            assert(ST->getTypeAtIndex((unsigned)0) == ST->getTypeAtIndex(i));
          }
          if (ST->getTypeAtIndex((unsigned)0)->isFloatingPointTy())
            updateAnalysis(
                call.getArgOperand(i),
                TypeTree(ConcreteType(
                             ST->getTypeAtIndex((unsigned)0)->getScalarType()))
                    .Only(-1, &call),
                &call);
          else if (ST->getTypeAtIndex((unsigned)0)->isIntegerTy()) {
            updateAnalysis(call.getArgOperand(i),
                           TypeTree(BaseType::Integer).Only(-1, &call), &call);
          } else {
            llvm::errs() << *T << " - " << call << "\n";
            llvm_unreachable("Unknown type for libm");
          }
        } else if (auto AT = dyn_cast<ArrayType>(T)) {
          assert(AT->getNumElements() >= 1);
          if (AT->getElementType()->isFloatingPointTy())
            updateAnalysis(
                call.getArgOperand(i),
                TypeTree(ConcreteType(AT->getElementType()->getScalarType()))
                    .Only(-1, &call),
                &call);
          else if (AT->getElementType()->isIntegerTy()) {
            updateAnalysis(call.getArgOperand(i),
                           TypeTree(BaseType::Integer).Only(-1, &call), &call);
          } else {
            llvm::errs() << *T << " - " << call << "\n";
            llvm_unreachable("Unknown type for libm");
          }
        } else {
          llvm::errs() << *T << " - " << call << "\n";
          llvm_unreachable("Unknown type for libm");
        }
      }
      Type *T = call.getType();
      if (T->isFloatingPointTy()) {
        updateAnalysis(&call,
                       TypeTree(ConcreteType(call.getType()->getScalarType()))
                           .Only(-1, &call),
                       &call);
      } else if (T->isIntegerTy()) {
        updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call),
                       &call);
      } else if (T->isVoidTy()) {
      } else if (auto ST = dyn_cast<StructType>(T)) {
        assert(ST->getNumElements() >= 1);
        TypeTree TT;
        auto &DL = call.getParent()->getParent()->getParent()->getDataLayout();
        for (size_t i = 0; i < ST->getNumElements(); ++i) {
          auto T = ST->getTypeAtIndex(i);
          ConcreteType CT(BaseType::Unknown);

          Value *vec[2] = {
              ConstantInt::get(Type::getInt64Ty(call.getContext()), 0),
              ConstantInt::get(Type::getInt32Ty(call.getContext()), i)};
          auto ud = UndefValue::get(getUnqual(ST));
          auto g2 = GetElementPtrInst::Create(ST, ud, vec);
          APInt ai(DL.getIndexSizeInBits(0), 0);
          g2->accumulateConstantOffset(DL, ai);
          delete g2;
          size_t Offset = ai.getZExtValue();

          size_t nextOffset;
          if (i + 1 == ST->getNumElements())
            nextOffset = (DL.getTypeSizeInBits(ST) + 7) / 8;
          else {
            Value *vec[2] = {
                ConstantInt::get(Type::getInt64Ty(call.getContext()), 0),
                ConstantInt::get(Type::getInt32Ty(call.getContext()), i + 1)};
            auto ud = UndefValue::get(getUnqual(ST));
            auto g2 = GetElementPtrInst::Create(ST, ud, vec);
            APInt ai(DL.getIndexSizeInBits(0), 0);
            g2->accumulateConstantOffset(DL, ai);
            delete g2;
            nextOffset = ai.getZExtValue();
          }

          if (T->isFloatingPointTy()) {
            CT = T;
          } else if (T->isIntegerTy()) {
            CT = BaseType::Integer;
          }
          if (CT != BaseType::Unknown) {
            TypeTree mid = TypeTree(CT).Only(-1, &call);
            TT |= mid.ShiftIndices(DL, /*init offset*/ 0,
                                   /*maxSize*/ nextOffset - Offset,
                                   /*addOffset*/ Offset);
          }
        }
        auto Size = (DL.getTypeSizeInBits(ST) + 7) / 8;
        TT.CanonicalizeInPlace(Size, DL);
        updateAnalysis(&call, TT, &call);
      } else if (auto AT = dyn_cast<ArrayType>(T)) {
        assert(AT->getNumElements() >= 1);
        if (AT->getElementType()->isFloatingPointTy())
          updateAnalysis(
              &call,
              TypeTree(ConcreteType(AT->getElementType()->getScalarType()))
                  .Only(-1, &call),
              &call);
        else {
          llvm::errs() << *T << " - " << call << "\n";
          llvm_unreachable("Unknown type for libm");
        }
      } else {
        llvm::errs() << *T << " - " << call << "\n";
        llvm_unreachable("Unknown type for libm");
      }
      return;
    }
    if (funcName == "__lgamma_r_finite") {
      updateAnalysis(
          call.getArgOperand(0),
          TypeTree(ConcreteType(Type::getDoubleTy(call.getContext())))
              .Only(-1, &call),
          &call);
      updateAnalysis(call.getArgOperand(1),
                     TypeTree(BaseType::Integer).Only(0, &call).Only(-1, &call),
                     &call);
      updateAnalysis(
          &call,
          TypeTree(ConcreteType(Type::getDoubleTy(call.getContext())))
              .Only(-1, &call),
          &call);
    }
    if (funcName == "__fd_sincos_1" || funcName == "__fd_sincos_1f" ||
        funcName == "__fd_sincos_1l") {
      updateAnalysis(call.getArgOperand(0),
                     TypeTree(ConcreteType(call.getArgOperand(0)->getType()))
                         .Only(-1, &call),
                     &call);
      updateAnalysis(&call,
                     TypeTree(ConcreteType(call.getArgOperand(0)->getType()))
                         .Only(-1, &call),
                     &call);
    }
    if (funcName == "frexp" || funcName == "frexpf" || funcName == "frexpl") {

      updateAnalysis(
          &call, TypeTree(ConcreteType(call.getType())).Only(-1, &call), &call);
      updateAnalysis(call.getOperand(0),
                     TypeTree(ConcreteType(call.getType())).Only(-1, &call),
                     &call);
      TypeTree ival(BaseType::Pointer);
      size_t objSize = 1;

#if LLVM_VERSION_MAJOR < 17
      auto &DL = fntypeinfo.Function->getParent()->getDataLayout();
      objSize = DL.getTypeSizeInBits(
                    call.getOperand(1)->getType()->getPointerElementType()) /
                8;
#endif
      for (size_t i = 0; i < objSize; ++i) {
        ival.insert({(int)i}, BaseType::Integer);
      }
      updateAnalysis(call.getOperand(1), ival.Only(-1, &call), &call);
      return;
    }

    if (funcName == "__cxa_guard_acquire" || funcName == "printf" ||
        funcName == "vprintf" || funcName == "puts" || funcName == "fputc" ||
        funcName == "fprintf") {
      updateAnalysis(&call, TypeTree(BaseType::Integer).Only(-1, &call), &call);
    }

    if (dontAnalyze(funcName))
      return;

    if (!ci->empty() && !hasMetadata(ci, "enzyme_gradient") &&
        !hasMetadata(ci, "enzyme_derivative")) {
      visitIPOCall(call, *ci);
    }
  }
}

TypeTree TypeAnalyzer::getReturnAnalysis() {
  bool set = false;
  TypeTree vd;
  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (auto &inst : BB) {
      if (auto ri = dyn_cast<ReturnInst>(&inst)) {
        if (auto rv = ri->getReturnValue()) {
          if (set == false) {
            set = true;
            vd = getAnalysis(rv);
            continue;
          }
          vd &= getAnalysis(rv);
          // TODO insert the selectinst anything propagation here
          // however this needs to be done simultaneously with preventing
          // anything from propagating up through the return value (if there
          // are multiple possible returns)
        }
      }
    }
  }
  return vd;
}

/// Helper function that calculates whether a given value must only be
/// an integer and cannot be cast/stored to be used as a ptr/integer
bool TypeAnalyzer::mustRemainInteger(Value *val, bool *returned) {
  std::map<Value *, std::pair<bool, bool>> &seen = mriseen;
  const DataLayout &DL = fntypeinfo.Function->getParent()->getDataLayout();
  if (seen.find(val) != seen.end()) {
    if (returned)
      *returned |= seen[val].second;
    return seen[val].first;
  }
  seen[val] = std::make_pair(true, false);
  for (auto u : val->users()) {
    if (auto SI = dyn_cast<StoreInst>(u)) {
      if (parseTBAA(*SI, DL, MST).Inner0().isIntegral())
        continue;
      seen[val].first = false;
      continue;
    }
    if (isa<CastInst>(u)) {
      if (!u->getType()->isIntOrIntVectorTy()) {
        seen[val].first = false;
        continue;
      } else if (!mustRemainInteger(u, returned)) {
        seen[val].first = false;
        seen[val].second |= seen[u].second;
        continue;
      } else
        continue;
    }
    if (isa<BinaryOperator>(u) || isa<IntrinsicInst>(u) || isa<PHINode>(u) ||
#if LLVM_VERSION_MAJOR <= 17
        isa<UDivOperator>(u) || isa<SDivOperator>(u) ||
#endif
        isa<LShrOperator>(u) || isa<AShrOperator>(u) || isa<AddOperator>(u) ||
        isa<MulOperator>(u) || isa<ShlOperator>(u)) {
      if (!mustRemainInteger(u, returned)) {
        seen[val].first = false;
        seen[val].second |= seen[u].second;
      }
      continue;
    }
    if (auto gep = dyn_cast<GetElementPtrInst>(u)) {
      if (gep->isInBounds() && gep->getPointerOperand() != val) {
        continue;
      }
    }
    if (returned && isa<ReturnInst>(u)) {
      *returned = true;
      seen[val].second = true;
      continue;
    }
    if (auto CI = dyn_cast<CallBase>(u)) {
      if (auto F = CI->getCalledFunction()) {
        if (!F->empty()) {
          int argnum = 0;
          bool subreturned = false;
          for (auto &arg : F->args()) {
            if (CI->getArgOperand(argnum) == val &&
                !mustRemainInteger(&arg, &subreturned)) {
              seen[val].first = false;
              seen[val].second |= seen[&arg].second;
              continue;
            }
            ++argnum;
          }
          if (subreturned && !mustRemainInteger(CI, returned)) {
            seen[val].first = false;
            seen[val].second |= seen[CI].second;
            continue;
          }
          continue;
        }
      }
    }
    if (isa<CmpInst>(u))
      continue;
    seen[val].first = false;
    seen[val].second = true;
  }
  if (returned && seen[val].second)
    *returned = true;
  return seen[val].first;
}

FnTypeInfo TypeAnalyzer::getCallInfo(CallBase &call, Function &fn) {
  FnTypeInfo typeInfo(&fn);

  size_t argnum = 0;
  for (auto &arg : fn.args()) {
    if (argnum >= call.arg_size()) {
      typeInfo.Arguments.insert(
          std::pair<Argument *, TypeTree>(&arg, TypeTree()));
      std::set<int64_t> bounded;
      typeInfo.KnownValues.insert(
          std::pair<Argument *, std::set<int64_t>>(&arg, bounded));
      ++argnum;
      continue;
    }
    auto dt = getAnalysis(call.getArgOperand(argnum));
    if (arg.getType()->isIntOrIntVectorTy() &&
        dt.Inner0() == BaseType::Anything) {
      if (mustRemainInteger(&arg)) {
        dt = TypeTree(BaseType::Integer).Only(-1, &call);
      }
    }
    typeInfo.Arguments.insert(std::pair<Argument *, TypeTree>(&arg, dt));
    std::set<int64_t> bounded;
    for (auto v : fntypeinfo.knownIntegralValues(call.getArgOperand(argnum), DT,
                                                 intseen, SE)) {
      if (abs(v) > MaxIntOffset)
        continue;
      bounded.insert(v);
    }
    typeInfo.KnownValues.insert(
        std::pair<Argument *, std::set<int64_t>>(&arg, bounded));
    ++argnum;
  }

  typeInfo.Return = getAnalysis(&call);
  return typeInfo;
}

void TypeAnalyzer::visitIPOCall(CallBase &call, Function &fn) {
#if LLVM_VERSION_MAJOR >= 14
  if (call.arg_size() != fn.getFunctionType()->getNumParams())
    return;
#else
  if (call.getNumArgOperands() != fn.getFunctionType()->getNumParams())
    return;
#endif

  assert(fntypeinfo.KnownValues.size() ==
         fntypeinfo.Function->getFunctionType()->getNumParams());

  bool hasDown = direction & DOWN;
  bool hasUp = direction & UP;

  if (hasDown) {
    if (call.getType()->isVoidTy())
      hasDown = false;
    else {
      if (getAnalysis(&call).IsFullyDetermined())
        hasDown = false;
    }
  }
  if (hasUp) {
    bool unknown = false;
#if LLVM_VERSION_MAJOR >= 14
    for (auto &arg : call.args())
#else
    for (auto &arg : call.arg_operands())
#endif
    {
      if (isa<ConstantData>(arg))
        continue;
      if (!getAnalysis(arg).IsFullyDetermined()) {
        unknown = true;
        break;
      }
    }
    if (!unknown)
      hasUp = false;
  }

  // Fast path where all information has already been derived
  if (!hasUp && !hasDown)
    return;

  FnTypeInfo typeInfo = getCallInfo(call, fn);
  typeInfo = preventTypeAnalysisLoops(typeInfo, call.getParent()->getParent());

  if (EnzymePrintType) {
    llvm::errs() << " starting IPO of ";
    call.print(llvm::errs(), *MST);
    llvm::errs() << "\n";
  }

  TypeResults STR = interprocedural.analyzeFunction(typeInfo);

  if (EnzymePrintType) {
    llvm::errs() << " ending IPO of ";
    call.print(llvm::errs(), *MST);
    llvm::errs() << "\n";
  }

  if (hasUp) {
    auto a = fn.arg_begin();
#if LLVM_VERSION_MAJOR >= 14
    for (auto &arg : call.args())
#else
    for (auto &arg : call.arg_operands())
#endif
    {
      auto dt = STR.query(a);
      if (EnzymePrintType) {
        llvm::errs() << " updating ";
        arg->print(llvm::errs(), *MST);
        llvm::errs() << " = " << dt.str() << "  via IPO of ";
        call.print(llvm::errs(), *MST);
        llvm::errs() << " arg ";
        a->print(llvm::errs(), *MST);
        llvm::errs() << "\n";
      }
      updateAnalysis(arg, dt, &call);
      ++a;
    }
  }

  if (hasDown) {
    TypeTree vd = STR.getReturnAnalysis();
    if (call.getType()->isIntOrIntVectorTy() &&
        vd.Inner0() == BaseType::Anything) {
      bool returned = false;
      if (mustRemainInteger(&call, &returned) && !returned) {
        vd = TypeTree(BaseType::Integer).Only(-1, &call);
      }
    }
    updateAnalysis(&call, vd, &call);
  }
}

TypeResults TypeAnalysis::analyzeFunction(const FnTypeInfo &fn) {
  assert(fn.KnownValues.size() ==
         fn.Function->getFunctionType()->getNumParams());
  assert(fn.Function);
  auto found = analyzedFunctions.find(fn);
  if (found != analyzedFunctions.end()) {
    auto &analysis = *found->second;
    if (analysis.fntypeinfo.Function != fn.Function) {
      llvm::errs() << " queryFunc: " << *fn.Function << "\n";
      llvm::errs() << " analysisFunc: " << *analysis.fntypeinfo.Function
                   << "\n";
    }
    assert(analysis.fntypeinfo.Function == fn.Function);

    return TypeResults(analysis);
  }

  if (fn.Function->empty())
    return TypeResults(nullptr);

  auto res = analyzedFunctions.emplace(fn, new TypeAnalyzer(fn, *this));
  auto &analysis = *res.first->second;

  if (EnzymePrintType) {
    llvm::errs() << "analyzing function " << fn.Function->getName() << "\n";
    for (auto &pair : fn.Arguments) {
      llvm::errs() << " + knowndata: ";
      pair.first->print(llvm::errs(), *analysis.MST);
      llvm::errs() << " : " << pair.second.str();
      auto found = fn.KnownValues.find(pair.first);
      if (found != fn.KnownValues.end()) {
        llvm::errs() << " - " << to_string(found->second);
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << " + retdata: " << fn.Return.str() << "\n";
  }

  analysis.prepareArgs();
  if (RustTypeRules) {
    analysis.considerRustDebugInfo();
  }
  analysis.considerTBAA();
  analysis.run();

  if (analysis.fntypeinfo.Function != fn.Function) {
    llvm::errs() << " queryFunc: " << *fn.Function << "\n";
    llvm::errs() << " analysisFunc: " << *analysis.fntypeinfo.Function << "\n";
  }
  assert(analysis.fntypeinfo.Function == fn.Function);

  {
    auto &analysis = *analyzedFunctions.find(fn)->second;
    if (analysis.fntypeinfo.Function != fn.Function) {
      llvm::errs() << " queryFunc: " << *fn.Function << "\n";
      llvm::errs() << " analysisFunc: " << *analysis.fntypeinfo.Function
                   << "\n";
    }
    assert(analysis.fntypeinfo.Function == fn.Function);
  }

  // Store the steady state result (if changed) to avoid
  // a second analysis later.
  analyzedFunctions.emplace(TypeResults(analysis).getAnalyzedTypeInfo(),
                            res.first->second);

  return TypeResults(analysis);
}

TypeResults::TypeResults(TypeAnalyzer &analyzer) : analyzer(&analyzer) {}
TypeResults::TypeResults(std::nullptr_t) : analyzer(nullptr) {}

FnTypeInfo TypeResults::getAnalyzedTypeInfo() const {
  FnTypeInfo res(analyzer->fntypeinfo.Function);
  for (auto &arg : analyzer->fntypeinfo.Function->args()) {
    res.Arguments.insert(std::pair<Argument *, TypeTree>(&arg, query(&arg)));
  }
  res.Return = getReturnAnalysis();
  res.KnownValues = analyzer->fntypeinfo.KnownValues;
  return res;
}

FnTypeInfo TypeResults::getCallInfo(CallBase &CI, Function &fn) const {
  return analyzer->getCallInfo(CI, fn);
}

TypeTree TypeResults::query(Value *val) const {
#ifndef NDEBUG
  if (auto inst = dyn_cast<Instruction>(val)) {
    assert(inst->getParent()->getParent() == analyzer->fntypeinfo.Function);
  }
  if (auto arg = dyn_cast<Argument>(val)) {
    assert(arg->getParent() == analyzer->fntypeinfo.Function);
  }
#endif
  return analyzer->getAnalysis(val);
}

// Returns last non-padding/alignment location of the corresponding subtype T.
size_t skippedBytes(SmallSet<size_t, 8> &offs, Type *T, const DataLayout &DL,
                    size_t offset = 0) {
  auto ST = dyn_cast<StructType>(T);
  if (!ST)
    return (DL.getTypeSizeInBits(T) + 7) / 8;

  auto SL = DL.getStructLayout(ST);
  size_t prevOff = 0;
  for (size_t idx = 0; idx < ST->getNumElements(); idx++) {
    auto off = SL->getElementOffset(idx);
    if (off > prevOff)
      for (size_t i = prevOff; i < off; i++)
        offs.insert(offset + i);
    size_t subSize = skippedBytes(offs, ST->getElementType(idx), DL, prevOff);
    prevOff = off + subSize;
  }
  return prevOff;
}

bool TypeResults::allFloat(Value *val) const {
  assert(val);
  assert(val->getType());
  auto q = query(val);
  auto dt = q[{-1}];
  if (dt != BaseType::Anything && dt != BaseType::Unknown)
    return dt.isFloat();

  if (val->getType()->isTokenTy() || val->getType()->isVoidTy())
    return false;
  auto &dl = analyzer->fntypeinfo.Function->getParent()->getDataLayout();
  SmallSet<size_t, 8> offs;
  size_t ObjSize = skippedBytes(offs, val->getType(), dl);

  for (size_t i = 0; i < ObjSize;) {
    dt = q[{(int)i}];
    if (auto FT = dt.isFloat()) {
      i += (dl.getTypeSizeInBits(FT) + 7) / 8;
      continue;
    }
    if (offs.count(i)) {
      i++;
      continue;
    }
    return false;
  }
  return true;
}

bool TypeResults::anyFloat(Value *val, bool anythingIsFloat) const {
  assert(val);
  assert(val->getType());
  auto q = query(val);
  auto dt = q[{-1}];
  if (!anythingIsFloat && dt == BaseType::Anything)
    return false;
  if (dt != BaseType::Anything && dt != BaseType::Unknown)
    return dt.isFloat();

  if (val->getType()->isTokenTy() || val->getType()->isVoidTy())
    return false;
  auto &dl = analyzer->fntypeinfo.Function->getParent()->getDataLayout();
  SmallSet<size_t, 8> offs;
  size_t ObjSize = skippedBytes(offs, val->getType(), dl);

  for (size_t i = 0; i < ObjSize;) {
    dt = q[{(int)i}];
    if (dt == BaseType::Integer) {
      i++;
      continue;
    }
    if (!anythingIsFloat && dt == BaseType::Integer) {
      i++;
      continue;
    }
    if (dt == BaseType::Pointer) {
      i += dl.getPointerSize(0);
      continue;
    }
    if (offs.count(i)) {
      i++;
      continue;
    }
    return true;
  }
  return false;
}

bool TypeResults::anyPointer(Value *val) const {
  assert(val);
  assert(val->getType());
  auto q = query(val);
  auto dt = q[{-1}];
  if (dt != BaseType::Anything && dt != BaseType::Unknown)
    return dt == BaseType::Pointer;
  if (val->getType()->isTokenTy() || val->getType()->isVoidTy())
    return false;

  auto &dl = analyzer->fntypeinfo.Function->getParent()->getDataLayout();
  SmallSet<size_t, 8> offs;
  size_t ObjSize = skippedBytes(offs, val->getType(), dl);

  for (size_t i = 0; i < ObjSize;) {
    dt = q[{(int)i}];
    if (dt == BaseType::Integer) {
      i++;
      continue;
    }
    if (auto FT = dt.isFloat()) {
      i += (dl.getTypeSizeInBits(FT) + 7) / 8;
      continue;
    }
    if (offs.count(i)) {
      i++;
      continue;
    }
    return true;
  }
  return false;
}

void TypeResults::dump(llvm::raw_ostream &ss) const { analyzer->dump(ss); }

ConcreteType TypeResults::intType(size_t num, Value *val, bool errIfNotFound,
                                  bool pointerIntSame) const {
  assert(val);
  assert(val->getType());
  auto q = query(val);
  auto dt = q[{0}];
  /*
  size_t ObjSize = 1;
  if (val->getType()->isSized())
    ObjSize = (fn.Function->getParent()->getDataLayout().getTypeSizeInBits(
        val->getType()) +7) / 8;
  */
  dt.orIn(q[{-1}], pointerIntSame);
  for (size_t i = 1; i < num; ++i) {
    dt.orIn(q[{(int)i}], pointerIntSame);
  }

  if (errIfNotFound && (!dt.isKnown() || dt == BaseType::Anything)) {
    if (auto inst = dyn_cast<Instruction>(val)) {
      llvm::errs() << *inst->getParent()->getParent()->getParent() << "\n";
      llvm::errs() << *inst->getParent()->getParent() << "\n";
      for (auto &pair : analyzer->analysis) {
        llvm::errs() << "val: " << *pair.first << " - " << pair.second.str()
                     << "\n";
      }
    }
    llvm::errs() << "could not deduce type of integer " << *val << "\n";
    assert(0 && "could not deduce type of integer");
  }
  return dt;
}

Type *TypeResults::addingType(size_t num, Value *val, size_t start) const {
  assert(val);
  assert(val->getType());
  auto q = query(val);
  Type *ty = q[{-1}].isFloat();
  for (size_t i = start; i < num; ++i) {
    auto ty2 = q[{(int)i}].isFloat();
    if (ty) {
      if (ty2)
        assert(ty == ty2);
    } else {
      ty = ty2;
    }
  }
  return ty;
}

ConcreteType TypeResults::firstPointer(size_t num, Value *val, Instruction *I,
                                       bool errIfNotFound,
                                       bool pointerIntSame) const {
  assert(val);
  assert(val->getType());
  auto q = query(val).Data0();
  if (!(val->getType()->isPointerTy() || q[{}] == BaseType::Pointer)) {
    llvm::errs() << *analyzer->fntypeinfo.Function << "\n";
    dump();
    llvm::errs() << "val: " << *val << "\n";
  }
  assert(val->getType()->isPointerTy() || q[{}] == BaseType::Pointer);

  auto dt = q[{-1}];
  for (size_t i = 0; i < num; ++i) {
    bool Legal = true;
    dt.checkedOrIn(q[{(int)i}], pointerIntSame, Legal);
    if (!Legal) {
      std::string str;
      raw_string_ostream ss(str);
      ss << "Illegal firstPointer, num: " << num << " q: " << q.str() << "\n";
      ss << " at " << *val << " from " << *I << "\n";
      if (CustomErrorHandler) {
        CustomErrorHandler(str.c_str(), wrap(I), ErrorType::IllegalFirstPointer,
                           &analyzer, nullptr, nullptr);
      }
      llvm::errs() << ss.str() << "\n";
      llvm_unreachable("Illegal firstPointer");
    }
  }

  if (errIfNotFound && (!dt.isKnown() || dt == BaseType::Anything)) {
    auto &res = *analyzer;
    if (auto inst = dyn_cast<Instruction>(val)) {
      llvm::errs() << *inst->getParent()->getParent()->getParent() << "\n";
      llvm::errs() << *inst->getParent()->getParent() << "\n";
      for (auto &pair : res.analysis) {
        if (auto in = dyn_cast<Instruction>(pair.first)) {
          if (in->getParent()->getParent() != inst->getParent()->getParent()) {
            llvm::errs() << "inf: " << *in->getParent()->getParent() << "\n";
            llvm::errs() << "instf: " << *inst->getParent()->getParent()
                         << "\n";
            llvm::errs() << "in: " << *in << "\n";
            llvm::errs() << "inst: " << *inst << "\n";
          }
          assert(in->getParent()->getParent() ==
                 inst->getParent()->getParent());
        }
        llvm::errs() << "val: " << *pair.first << " - " << pair.second.str()
                     << " int: " +
                            to_string(res.knownIntegralValues(pair.first))
                     << "\n";
      }
    }
    if (auto arg = dyn_cast<Argument>(val)) {
      llvm::errs() << *arg->getParent() << "\n";
      for (auto &pair : res.analysis) {
#ifndef NDEBUG
        if (auto in = dyn_cast<Instruction>(pair.first))
          assert(in->getParent()->getParent() == arg->getParent());
#endif
        llvm::errs() << "val: " << *pair.first << " - " << pair.second.str()
                     << " int: " +
                            to_string(res.knownIntegralValues(pair.first))
                     << "\n";
      }
    }
    llvm::errs() << "fn: " << *analyzer->fntypeinfo.Function << "\n";
    dump();
    llvm::errs() << "could not deduce type of integer " << *val
                 << " num:" << num << " q:" << q.str() << " \n";

    llvm::DiagnosticLocation loc =
        analyzer->fntypeinfo.Function->getSubprogram();
    Instruction *codeLoc =
        &*analyzer->fntypeinfo.Function->getEntryBlock().begin();
    if (auto inst = dyn_cast<Instruction>(val)) {
      loc = inst->getDebugLoc();
      codeLoc = inst;
    }
    EmitFailure("CannotDeduceType", loc, codeLoc,
                "failed to deduce type of value ", *val);

    assert(0 && "could not deduce type of integer");
  }
  return dt;
}

/// Parse the debug info generated by rustc and retrieve useful type info if
/// possible
void TypeAnalyzer::considerRustDebugInfo() {
  DataLayout DL = fntypeinfo.Function->getParent()->getDataLayout();
  for (BasicBlock &BB : *fntypeinfo.Function) {
    for (Instruction &I : BB) {
      if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(&I)) {
        TypeTree TT = parseDIType(*DDI, DL);
        if (!TT.isKnown()) {
          continue;
        }
        TT |= TypeTree(BaseType::Pointer);
        updateAnalysis(DDI->getAddress(), TT.Only(-1, &I), DDI);
      }
    }
  }
}

TypeTree defaultTypeTreeForLLVM(llvm::Type *ET, llvm::Instruction *I,
                                bool intIsPointer) {
  if (ET->isIntOrIntVectorTy()) {
    if (intIsPointer)
      return TypeTree(BaseType::Pointer).Only(-1, I);
    else
      return TypeTree(BaseType::Integer).Only(-1, I);
  }
  if (ET->isFPOrFPVectorTy()) {
    return TypeTree(ConcreteType(ET->getScalarType())).Only(-1, I);
  }
  if (ET->isPointerTy()) {
    return TypeTree(BaseType::Pointer).Only(-1, I);
  }
  if (auto ST = dyn_cast<StructType>(ET)) {
    auto &DL = I->getParent()->getParent()->getParent()->getDataLayout();

    TypeTree Out;

    for (size_t i = 0; i < ST->getNumElements(); i++) {
      auto SubT =
          defaultTypeTreeForLLVM(ST->getElementType(i), I, intIsPointer);
      Value *vec[2] = {
          ConstantInt::get(Type::getInt64Ty(I->getContext()), 0),
          ConstantInt::get(Type::getInt32Ty(I->getContext()), i),
      };
      auto g2 =
          GetElementPtrInst::Create(ST, UndefValue::get(getUnqual(ST)), vec);
      APInt ai(DL.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
      g2->accumulateConstantOffset(DL, ai);
      // Using destructor rather than eraseFromParent
      //   as g2 has no parent
      delete g2;

      auto size = (DL.getTypeSizeInBits(ST->getElementType(i)) + 7) / 8;
      int Off = (int)ai.getLimitedValue();
      Out |= SubT.ShiftIndices(DL, 0, size, Off);
    }
    return Out;
  }
  if (auto AT = dyn_cast<ArrayType>(ET)) {
    auto SubT = defaultTypeTreeForLLVM(AT->getElementType(), I, intIsPointer);
    auto &DL = I->getParent()->getParent()->getParent()->getDataLayout();

    TypeTree Out;
    for (size_t i = 0; i < AT->getNumElements(); i++) {
      Value *vec[2] = {
          ConstantInt::get(Type::getInt64Ty(I->getContext()), 0),
          ConstantInt::get(Type::getInt32Ty(I->getContext()), i),
      };
      auto g2 =
          GetElementPtrInst::Create(AT, UndefValue::get(getUnqual(AT)), vec);
      APInt ai(DL.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
      g2->accumulateConstantOffset(DL, ai);
      // Using destructor rather than eraseFromParent
      //   as g2 has no parent
      delete g2;

      int Off = (int)ai.getLimitedValue();
      auto size = (DL.getTypeSizeInBits(AT->getElementType()) + 7) / 8;
      Out |= SubT.ShiftIndices(DL, 0, size, Off);
    }
    return Out;
  }
  if (auto AT = dyn_cast<VectorType>(ET)) {
#if LLVM_VERSION_MAJOR >= 12
    assert(!AT->getElementCount().isScalable());
    size_t numElems = AT->getElementCount().getKnownMinValue();
#else
    size_t numElems = AT->getNumElements();
#endif
    auto SubT = defaultTypeTreeForLLVM(AT->getElementType(), I, intIsPointer);
    auto &DL = I->getParent()->getParent()->getParent()->getDataLayout();

    TypeTree Out;
    for (size_t i = 0; i < numElems; i++) {
      Value *vec[2] = {
          ConstantInt::get(Type::getInt64Ty(I->getContext()), 0),
          ConstantInt::get(Type::getInt32Ty(I->getContext()), i),
      };
      auto g2 =
          GetElementPtrInst::Create(AT, UndefValue::get(getUnqual(AT)), vec);
      APInt ai(DL.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
      g2->accumulateConstantOffset(DL, ai);
      // Using destructor rather than eraseFromParent
      //   as g2 has no parent
      delete g2;

      int Off = (int)ai.getLimitedValue();
      auto size = (DL.getTypeSizeInBits(AT->getElementType()) + 7) / 8;
      Out |= SubT.ShiftIndices(DL, 0, size, Off);
    }
    return Out;
  }
  // Unhandled/unknown Type
  llvm::errs() << "Error Unknown Type: " << *ET << "\n";
  assert(0 && "Error Unknown Type: ");
  llvm_unreachable("Error Unknown Type: ");
  // return TypeTree();
}

Function *TypeResults::getFunction() const {
  return analyzer->fntypeinfo.Function;
}

TypeTree TypeResults::getReturnAnalysis() const {
  return analyzer->getReturnAnalysis();
}

std::set<int64_t> TypeResults::knownIntegralValues(Value *val) const {
  return analyzer->knownIntegralValues(val);
}

std::set<int64_t> TypeAnalyzer::knownIntegralValues(Value *val) {
  return fntypeinfo.knownIntegralValues(val, DT, intseen, SE);
}

void TypeAnalysis::clear() { analyzedFunctions.clear(); }

FnTypeInfo preventTypeAnalysisLoops(const FnTypeInfo &oldTypeInfo_,
                                    llvm::Function *todiff) {
  FnTypeInfo oldTypeInfo = oldTypeInfo_;
  for (auto &pair : oldTypeInfo.KnownValues) {
    if (pair.second.size() != 0) {
      bool recursiveUse = false;
      std::set<std::pair<Value *, Value *>> seen;
      SetVector<std::pair<Value *, Value *>> todo;
      for (auto user : pair.first->users())
        todo.insert(std::make_pair(user, pair.first));
      while (todo.size()) {
        auto spair = todo.pop_back_val();
        if (seen.count(spair))
          continue;
        seen.insert(spair);
        auto [v, prev] = spair;
        if (isa<BinaryOperator>(v) || isa<PHINode>(v) || isa<Argument>(v)) {
          for (auto user : v->users())
            todo.insert(std::make_pair(user, v));
          continue;
        }
        if (auto ci = dyn_cast<CallBase>(v)) {
          if (ci->getCalledFunction() == todiff &&
              ci->getArgOperand(pair.first->getArgNo()) == prev) {
            if (prev == pair.first)
              continue;
            recursiveUse = true;
            break;
          }
        }
      }
      if (recursiveUse) {
        pair.second.clear();
      }
    }
  }
  return oldTypeInfo;
}
