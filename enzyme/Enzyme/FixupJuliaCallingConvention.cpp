#include "CApi.h"
#include "FunctionUtils.h"
#include "GradientUtils.h"
#include "Utils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define addAttribute addAttributeAtIndex
#define removeAttribute removeAttributeAtIndex
#define getAttribute getAttributeAtIndex
#define hasAttribute hasAttributeAtIndex

#if LLVM_VERSION_MAJOR >= 17
#include "llvm/TargetParser/Triple.h"
#endif

using namespace llvm;

extern bool
DetectPointerArgOfFn(llvm::Function &F,
                     llvm::SmallPtrSetImpl<llvm::Function *> &calls_todo);

bool needsReRooting(llvm::Argument *arg, bool &anyJLStore,
                    llvm::Type *SRetType = nullptr) {
  auto Attrs = arg->getParent()->getAttributes();

  if (!SRetType)
    SRetType = convertSRetTypeFromString(
        Attrs
            .getAttribute(AttributeList::FirstArgIndex + arg->getArgNo(),
                          "enzyme_sret")
            .getValueAsString(),
        &arg->getContext());

  CountTrackedPointers tracked(SRetType);
  if (tracked.count == 0) {
    return false;
  }

  bool hasReturnRootingAfterArg = false;
  for (size_t i = arg->getArgNo() + 1; i < arg->getParent()->arg_size(); i++) {
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           "enzymejl_returnRoots")) {
      hasReturnRootingAfterArg = true;
      break;
    }
  }

  // If there is no returnRoots, we _must_ reroot the arg.
  if (!hasReturnRootingAfterArg) {
    return true;
  }

  SmallVector<Value *> storedValues;

  auto &DL = arg->getParent()->getParent()->getDataLayout();
  SmallVector<size_t> sret_offsets;
  {
    std::deque<std::pair<llvm::Type *, std::vector<unsigned>>> todo = {
        {SRetType, {}}};
    while (!todo.empty()) {
      auto cur = std::move(todo[0]);
      todo.pop_front();
      auto path = std::move(cur.second);
      auto ty = cur.first;

      if (auto PT = dyn_cast<PointerType>(ty)) {
        if (!isSpecialPtr(PT))
          continue;

        SmallVector<Constant *, 1> IdxList;
        IdxList.push_back(
            ConstantInt::get(Type::getInt64Ty(PT->getContext()), 0));

        for (auto v : path)
          IdxList.push_back(
              ConstantInt::get(Type::getInt32Ty(PT->getContext()), v));
        auto nullp = ConstantPointerNull::get(PointerType::getUnqual(SRetType));
        auto gep = ConstantExpr::getGetElementPtr(SRetType, nullp, IdxList);

        if (gep == ConstantPointerNull::get(PointerType::getUnqual(PT))) {
          sret_offsets.push_back(0);
          continue;
        }
#if LLVM_VERSION_MAJOR >= 20
        SmallMapVector<Value *, APInt, 4> VariableOffsets;
#else
        MapVector<Value *, APInt> VariableOffsets;
#endif
        auto width = DL.getPointerSize() * 8;
        APInt Offset(width, 0);
        bool success = collectOffset(cast<GEPOperator>(gep), DL, width,
                                     VariableOffsets, Offset);
        if (!success)
          llvm_unreachable("Illegal offset collection");
        sret_offsets.push_back(Offset.getZExtValue());
        continue;
      }

      if (auto AT = dyn_cast<ArrayType>(ty)) {
        for (size_t i = 0; i < AT->getNumElements(); i++) {
          std::vector<unsigned> path2(path);
          path2.push_back(i);
          todo.emplace_back(AT->getElementType(), path2);
        }
        continue;
      }

      if (auto VT = dyn_cast<VectorType>(ty)) {
        for (size_t i = 0; i < VT->getElementCount().getKnownMinValue(); i++) {
          std::vector<unsigned> path2(path);
          path2.push_back(i);
          todo.emplace_back(VT->getElementType(), path2);
        }
        continue;
      }

      if (auto ST = dyn_cast<StructType>(ty)) {
        for (size_t i = 0; i < ST->getNumElements(); i++) {
          std::vector<unsigned> path2(path);
          path2.push_back(i);
          todo.emplace_back(ST->getTypeAtIndex(i), path2);
        }
        continue;
      }
    }
  }

  bool legal = true;
  for (auto &&[I, cur, byteOffset] : findAllUsersOf(arg)) {
    assert(I->getParent()->getParent() == arg->getParent());

    if (isa<ICmpInst>(I)) {
      continue;
    }
    if (isa<LoadInst>(I)) {
      continue;
    }
    if (auto SI = dyn_cast<StoreInst>(I)) {
      assert(SI->getValueOperand() != cur);

      if (CountTrackedPointers(SI->getValueOperand()->getType()).count == 0)
        continue;

      storedValues.push_back(SI->getValueOperand());
      anyJLStore = true;
      continue;
    }

    if (isa<MemSetInst>(I))
      continue;

    if (auto MSI = dyn_cast<MemTransferInst>(I)) {
      if (auto Len = dyn_cast<ConstantInt>(MSI->getLength())) {
        bool mlegal = true;
        for (auto offset : sret_offsets) {
          if (byteOffset + Len->getSExtValue() <= offset)
            continue;
          if (offset + DL.getPointerSize() <= byteOffset)
            continue;
          mlegal = false;
          break;
        }
        if (mlegal)
          break;
      }
    }

    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "Unknown user of sret-like argument\n";
    CustomErrorHandler(ss.str().c_str(), wrap(I), ErrorType::GCRewrite,
                       wrap(cur), wrap(arg), nullptr);
    legal = false;
    anyJLStore = true;
    break;
  }

  if (legal) {
    while (!storedValues.empty()) {
      auto sv = storedValues.pop_back_val();
      if (auto I = dyn_cast<Instruction>(sv)) {
        assert(I->getParent()->getParent() == arg->getParent());
      }
      bool foundUse = false;
      for (auto &U : sv->uses()) {
        if (auto SI = dyn_cast<StoreInst>(U.getUser())) {
          if (SI->getValueOperand() == sv) {
            auto base = getBaseObject(SI->getPointerOperand());
            if (base == arg) {
              continue;
            }
            if (auto evi = dyn_cast<ExtractValueInst>(base)) {
              base = evi->getAggregateOperand();
            }
            if (auto arg2 = dyn_cast<Argument>(base)) {
              if (Attrs
                      .getAttribute(AttributeList::FirstArgIndex +
                                        arg2->getArgNo(),
                                    "enzymejl_returnRoots")
                      .isValid()) {
                foundUse = true;
                break;
              }
            }
          }
        }
      }
      if (!foundUse) {
        if (auto IVI = dyn_cast<InsertValueInst>(sv)) {
          CountTrackedPointers tracked(
              IVI->getInsertedValueOperand()->getType());
          if (tracked.count == 0) {
            storedValues.push_back(IVI->getAggregateOperand());
            continue;
          }
          if (isa<UndefValue>(IVI->getAggregateOperand()) ||
              isa<PoisonValue>(IVI->getAggregateOperand()) ||
              isa<ConstantAggregateZero>(IVI->getAggregateOperand())) {
            storedValues.push_back(IVI->getInsertedValueOperand());
            continue;
          }
          storedValues.push_back(IVI->getAggregateOperand());
          storedValues.push_back(IVI->getInsertedValueOperand());
          continue;
        }
        if (auto ST = dyn_cast<StructType>(sv->getType())) {
          bool legal = true;
          for (size_t i = 0; i < ST->getNumElements(); i++) {

            CountTrackedPointers tracked(ST->getElementType(i));
            if (tracked.count == 0) {
              continue;
            }
            std::map<std::vector<unsigned>, bool> paths_to_cover;
            {
              std::deque<std::pair<llvm::Type *, std::vector<unsigned>>> todo =
                  {{ST->getElementType(i), {}}};
              while (!todo.empty()) {
                auto cur = std::move(todo[0]);
                todo.pop_front();
                auto path = std::move(cur.second);
                auto ty = cur.first;

                if (auto PT = dyn_cast<PointerType>(ty)) {
                  if (isSpecialPtr(PT)) {
                    paths_to_cover[path] = false;
                  }
                  continue;
                }

                if (auto AT = dyn_cast<ArrayType>(ty)) {
                  for (size_t k = 0; k < AT->getNumElements(); k++) {
                    std::vector<unsigned> path2(path);
                    path2.push_back(k);
                    todo.emplace_back(AT->getElementType(), path2);
                  }
                  continue;
                }

                if (auto VT = dyn_cast<VectorType>(ty)) {
                  for (size_t k = 0;
                       k < VT->getElementCount().getKnownMinValue(); k++) {
                    std::vector<unsigned> path2(path);
                    path2.push_back(k);
                    todo.emplace_back(VT->getElementType(), path2);
                  }
                  continue;
                }

                if (auto ST2 = dyn_cast<StructType>(ty)) {
                  for (size_t k = 0; k < ST2->getNumElements(); k++) {
                    std::vector<unsigned> path2(path);
                    path2.push_back(k);
                    todo.emplace_back(ST2->getTypeAtIndex(k), path2);
                  }
                  continue;
                }
              }
            }

            for (auto u : sv->users()) {
              if (auto ev0 = dyn_cast<ExtractValueInst>(u)) {
                if (ev0->getIndices()[0] == i) {
                  std::vector<unsigned> extract_path;
                  for (size_t k = 1; k < ev0->getNumIndices(); ++k) {
                    extract_path.push_back(ev0->getIndices()[k]);
                  }
                  storedValues.push_back(ev0);

                  // Mark paths covered
                  for (auto &pair : paths_to_cover) {
                    const auto &p = pair.first;
                    bool match = true;
                    if (extract_path.size() > p.size()) {
                      match = false;
                    } else {
                      for (size_t idx = 0; idx < extract_path.size(); ++idx) {
                        if (extract_path[idx] != p[idx]) {
                          match = false;
                          break;
                        }
                      }
                    }
                    if (match) {
                      pair.second = true;
                    }
                  }
                }
              }
            }

            bool fullyCovered = true;
            for (const auto &pair : paths_to_cover) {
              if (!pair.second) {
                fullyCovered = false;
                break;
              }
            }

            if (!fullyCovered) {
              llvm::errs() << " failed to find extracted pointer for " << *sv
                           << " at index " << i << "\n";
              legal = false;
              break;
            }
          }
          if (legal) {
            continue;
          }
        }
        if (!isa<PointerType>(sv->getType()) ||
            !isSpecialPtr(cast<PointerType>(sv->getType()))) {
          llvm::errs() << " sf: " << *arg->getParent() << "\n";
          llvm::errs() << " arg: " << *arg << "\n";
          llvm::errs() << "Pointer of wrong type: " << *sv << "\n";
          assert(0);
        }

        {
          bool saw_bitcast = false;
          for (auto u : sv->users()) {
            if (auto ev0 = dyn_cast<CastInst>(u)) {
              auto t2 = ev0->getType();
              if (isa<PointerType>(t2) && isSpecialPtr(cast<PointerType>(t2))) {
                saw_bitcast = true;
                storedValues.push_back(ev0);
                break;
              }
            }
          }
          if (saw_bitcast)
            continue;
        }

        if (hasReturnRootingAfterArg) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << "Could not find use of stored value\n";
          ss << " sv: " << *sv << "\n";
          CustomErrorHandler(ss.str().c_str(), wrap(sv), ErrorType::GCRewrite,
                             nullptr, wrap(arg), nullptr);
        }
        legal = false;
        break;
      }
    }
  }

  return !legal;
}

// For a given enzymejl_returnRoots, which we assume is loaded after
// the call (and therefore is needed to be preserved), check whether
// there is an existing enzyme_sret for whom the roots could be assigned to,
// or if an additional sret argument is required.
// This is because count(sret_type) == returnRoots for the final merged type.
// As a result, there _must_ be a sret_type corresponding to the return root.
bool needsReReturning(llvm::Argument *arg, size_t &sret_idx,
                      std::map<size_t, size_t> &srets_without_stores) {
  auto Attrs = arg->getParent()->getAttributes();

  bool hasSRetBeforeArg = false;
  for (size_t i = 0; i < arg->getArgNo(); i++) {
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret")) {
      hasSRetBeforeArg = true;
      break;
    }
  }

  if (!hasSRetBeforeArg) {
    assert(srets_without_stores.size() == 0);
    return true;
  }

  if (srets_without_stores.size() == 0) {
    return true;
  }

  size_t subCount = convertRRootCountFromString(
      Attrs
          .getAttribute(AttributeList::FirstArgIndex + arg->getArgNo(),
                        "enzymejl_returnRoots")
          .getValueAsString());

  for (auto &pair : srets_without_stores) {
    if (pair.second == subCount) {
      sret_idx = pair.first;
      srets_without_stores.erase(sret_idx);
      return false;
    }
  }

  llvm_unreachable("Unsupported needsReRooting");
  return true;
}

static bool isOpaque(llvm::Type *T) {
#if LLVM_VERSION_MAJOR >= 20
  return T->isPointerTy();
#else
  return T->isOpaquePointerTy();
#endif
}

static void removeRange(std::vector<std::pair<uint64_t, uint64_t>> &ranges,
                        uint64_t start, uint64_t end) {
  std::vector<std::pair<uint64_t, uint64_t>> nextRanges;
  for (auto &range : ranges) {
    if (end <= range.first || start >= range.second) {
      nextRanges.push_back(range);
    } else {
      if (start > range.first) {
        nextRanges.push_back({range.first, start});
      }
      if (end < range.second) {
        nextRanges.push_back({end, range.second});
      }
    }
  }
  ranges = std::move(nextRanges);
}
static bool isReadOnlyNoCapture(Function *F, unsigned argNo) {
  return F->hasParamAttribute(argNo, Attribute::ReadOnly) &&
         F->getArg(argNo)->hasNoCaptureAttr();
}

static bool isGuaranteedToFullyWrite(Function *F, unsigned argNo, Type *T) {
  if (F->isDeclaration())
    return false;

  auto &DL = F->getParent()->getDataLayout();
  auto size = DL.getTypeAllocSize(T);

  std::vector<std::pair<uint64_t, uint64_t>> ranges = {{0, size}};
  std::vector<std::pair<Value *, uint64_t>> worklist = {{F->getArg(argNo), 0}};
  std::set<Value *> seen = {F->getArg(argNo)};

  PostDominatorTree PDT(*F);

  while (!worklist.empty()) {
    auto item = worklist.back();
    worklist.pop_back();
    Value *val = item.first;
    uint64_t offset = item.second;

    for (auto *U : val->users()) {
      if (auto *BI = dyn_cast<CastInst>(U)) {
        if (seen.insert(BI).second)
          worklist.push_back({BI, offset});
        continue;
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        APInt gepOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (GEP->accumulateConstantOffset(DL, gepOffset)) {
          if (seen.insert(GEP).second)
            worklist.push_back({GEP, offset + gepOffset.getZExtValue()});
          continue;
        }
      }

      if (auto *I = dyn_cast<Instruction>(U)) {
        if (I->getParent() != &F->getEntryBlock() &&
            !PDT.dominates(I->getParent(), &F->getEntryBlock()))
          continue;

        if (auto *SI = dyn_cast<StoreInst>(I)) {
          if (SI->getPointerOperand() == val) {
            auto storeSize =
                DL.getTypeAllocSize(SI->getValueOperand()->getType());
            removeRange(ranges, offset, offset + storeSize);
            if (ranges.empty())
              return true;
            continue;
          }
        }

        if (auto *MSI = dyn_cast<MemSetInst>(I)) {
          if (MSI->getDest() == val) {
            if (auto *CI = dyn_cast<ConstantInt>(MSI->getLength())) {
              removeRange(ranges, offset, offset + CI->getZExtValue());
              if (ranges.empty())
                return true;
              continue;
            }
          }
        }

        if (auto *MCI = dyn_cast<MemCpyInst>(I)) {
          if (MCI->getDest() == val) {
            if (auto *CI = dyn_cast<ConstantInt>(MCI->getLength())) {
              removeRange(ranges, offset, offset + CI->getZExtValue());
              if (ranges.empty())
                return true;
              continue;
            }
          }
        }
      }
    }
  }

  return ranges.empty();
}

// TODO, for sret/sret_v check if it actually stores the jlvalue_t's into the
// sret If so, confirm that those values are saved elsewhere in a returnroot
void EnzymeFixupJuliaCallingConvention(Function *F, bool sret_jlvalue) {
  if (F->empty())
    return;

  auto RT = F->getReturnType();
  std::set<size_t> srets;
  std::set<size_t> enzyme_srets;

  std::set<size_t> reroot_enzyme_srets;

  std::set<size_t> noroot_enzyme_srets;

  std::set<size_t> rroots;

  std::set<size_t> reret_roots;

  auto FT = F->getFunctionType();
  auto Attrs = F->getAttributes();

  std::map<size_t, size_t> selected_roots;

  // Map from the sret index to the number of stores, as unused
  std::map<size_t, size_t> srets_without_stores;

  for (size_t i = 0, end = FT->getNumParams(); i < end; i++) {
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           Attribute::StructRet))
      srets.insert(i);
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret")) {
      bool anyJLStore = false;
      enzyme_srets.insert(i);
      if (needsReRooting(F->getArg(i), anyJLStore)) {
        // Case 1: jlvalue_t's were stored into the sret, but were not stored
        // into an existing rooted argument.
        reroot_enzyme_srets.insert(i);
      } else if (anyJLStore) {
        // Case 2: jlvalue_t's were stored into the sret, and the were stored
        // into an existing rooted argument.
      } else {
        // Case 3: No jlvalue_t's were stored into the sret.
        llvm::Type *SRetType = convertSRetTypeFromString(
            Attrs.getAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret")
                .getValueAsString(),
            &F->getContext());
        if (auto count = CountTrackedPointers(SRetType).count) {
          srets_without_stores[i] = count;
          noroot_enzyme_srets.insert(i);
        }
      }
    }
    assert(
        !Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret_v"));

    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           "enzymejl_returnRoots")) {
      rroots.insert(i);
      size_t sret_idx;
      // Existing
      if (needsReReturning(F->getArg(i), sret_idx, srets_without_stores)) {
        reret_roots.insert(i);
      } else {
        selected_roots[i] = sret_idx;
      }
    }
    assert(!Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                               "enzymejl_returnRoots_v"));
  }

  // Regular julia function, needing no intervention
  if (srets.size() == 1) {
    assert(*srets.begin() == 0);
    assert(enzyme_srets.size() == 0);
    llvm::Type *SRetType = F->getParamStructRetType(0);
    CountTrackedPointers tracked(SRetType);

    // No jlvaluet to rewrite
    if (!tracked.count) {
      return;
    }

    bool anyJLStore = false;
    bool rerooting = needsReRooting(F->getArg(0), anyJLStore, SRetType);

    // We now assume we have an sret.
    // If it is properly rooted, we don't have any work to do
    if (rroots.size()) {
      assert(rroots.size() == 1);
      assert(*rroots.begin() == 1);
      // GVN is only powerful enough at LLVM 16+
      // (https://godbolt.org/z/ebY3exW9K)
#if LLVM_VERSION_MAJOR >= 16
      if (rerooting) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << "Illegal GC setup in which rerooting is required\n";
        ss << " + F: " << *F << "\n";
        CustomErrorHandler(s.c_str(), wrap(F), ErrorType::InternalError,
                           nullptr, nullptr, nullptr);
      }
      assert(!rerooting);
#endif

      size_t count = convertRRootCountFromString(
          Attrs
              .getAttribute(AttributeList::FirstArgIndex + 1,
                            "enzymejl_returnRoots")
              .getValueAsString());

      assert(count == tracked.count);
      return;
    }

    F->addParamAttr(0, Attribute::get(F->getContext(), "enzyme_sret",
                                      convertSRetTypeToString(SRetType)));
    Attrs = F->getAttributes();
    srets.clear();
    size_t i = 0;
    enzyme_srets.insert(i);
    if (rerooting) {
      reroot_enzyme_srets.insert(i);
    } else if (anyJLStore) {
    } else {
      if (auto count = CountTrackedPointers(SRetType).count) {
        srets_without_stores[i] = count;
        noroot_enzyme_srets.insert(i);
      }
    }
  } else if (srets.size() == 0 && enzyme_srets.size() == 0 &&
             rroots.size() == 0) {
    // No sret/rooting, no intervention needed.
    return;
  }

  // Number of additional roots, which contain actually no data at all.
  // Consider this additional rerooting of the sret, except this time
  // just fill it with 0's
  for (auto &pair : srets_without_stores) {
    assert(pair.second);
    reroot_enzyme_srets.insert(pair.first);
  }

  assert(srets.size() == 0);

  SmallVector<Type *, 1> Types;
  if (!RT->isVoidTy()) {
    Types.push_back(RT);
  }

  auto T_jlvalue = StructType::get(F->getContext(), {});
  auto T_prjlvalue = PointerType::get(T_jlvalue, AddressSpace::Tracked);

  size_t numRooting = RT->isVoidTy() ? 0 : CountTrackedPointers(RT).count;

  for (auto idx : enzyme_srets) {
    llvm::Type *SRetType = convertSRetTypeFromString(
        Attrs.getAttribute(AttributeList::FirstArgIndex + idx, "enzyme_sret")
            .getValueAsString(),
        &F->getContext());
#if LLVM_VERSION_MAJOR < 17
    if (F->getContext().supportsTypedPointers()) {
      auto T = FT->getParamType(idx)->getPointerElementType();
      if (T != SRetType) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << "Type mismatch in FixupJuliaCallingConvention:\n";
        ss << " + T: " << *T << "\n";
        ss << " + SRetType: " << *SRetType << "\n";
        EmitFailure("TypeMismatch", F->getSubprogram(), F, ss.str());
      }
    }
#endif
    Types.push_back(SRetType);
    if (reroot_enzyme_srets.count(idx)) {
      numRooting += CountTrackedPointers(SRetType).count;
    }
  }
  for (auto idx : rroots) {
    size_t count = convertRRootCountFromString(
        Attrs
            .getAttribute(AttributeList::FirstArgIndex + idx,
                          "enzymejl_returnRoots")
            .getValueAsString());
    auto T = ArrayType::get(T_prjlvalue, count);
#if LLVM_VERSION_MAJOR < 17
    if (F->getContext().supportsTypedPointers()) {
      auto NT = FT->getParamType(idx)->getPointerElementType();
      assert(NT == T);
    }
#endif
    if (reret_roots.count(idx)) {
      Types.push_back(T);
    }
    numRooting += count;
  }

  StructType *ST =
      Types.size() <= 1 ? nullptr : StructType::get(F->getContext(), Types);
  Type *sretTy = nullptr;
  if (Types.size())
    sretTy = Types.size() == 1 ? Types[0] : ST;

  ArrayType *roots_AT =
      numRooting ? ArrayType::get(T_prjlvalue, numRooting) : nullptr;

  if (sretTy) {
    CountTrackedPointers countF(sretTy);
    // If all fields of the sret struct are tracked pointers, the struct itself
    // acts as a root anchor on the caller's stack frame. In this scenario, we
    // do not allocate an additional explicit ReturnRoots array argument.
    if (countF.all) {
      roots_AT = nullptr;
      numRooting = 0;
      reroot_enzyme_srets.clear();
    } else if (countF.count) {
      if (!roots_AT) {
        llvm::errs() << " sretTy: " << *sretTy << "\n";
        llvm::errs() << " numRooting: " << numRooting << "\n";
        llvm::errs() << " tracked.count: " << countF.count << "\n";
      }
      assert(roots_AT);
      if (numRooting != countF.count) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << "Illegal GC setup in which numRooting (" << numRooting
           << ") != tracked.count (" << countF.count << ")\n";
        ss << " sretTy: " << *sretTy << "\n";
        ss << " Types.size(): " << Types.size() << "\n";
        for (size_t i = 0; i < Types.size(); i++) {
          ss << "    + Types[" << i << "] = " << *Types[i] << "\n";
        }
        ss << " F: " << *F << "\n";
        CustomErrorHandler(s.c_str(), wrap(F), ErrorType::InternalError,
                           nullptr, nullptr, nullptr);
      }
      assert(numRooting == countF.count);
    }
  }

  AttributeList NewAttrs;
  SmallVector<Type *, 1> types;
  size_t nexti = 0;
  if (sretTy) {
    types.push_back(getUnqual(sretTy));
    NewAttrs = NewAttrs.addAttribute(
        F->getContext(), AttributeList::FirstArgIndex + nexti,
        Attribute::get(F->getContext(), Attribute::StructRet, sretTy));
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FirstArgIndex + nexti,
                                     Attribute::NoAlias);
    nexti++;
  }
  if (roots_AT) {
    NewAttrs = NewAttrs.addAttribute(
        F->getContext(), AttributeList::FirstArgIndex + nexti,
        "enzymejl_returnRoots", std::to_string(numRooting));
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FirstArgIndex + nexti,
                                     Attribute::NoAlias);
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FirstArgIndex + nexti,
                                     Attribute::WriteOnly);
    types.push_back(getUnqual(roots_AT));
    nexti++;
  }

  for (size_t i = 0, end = FT->getNumParams(); i < end; i++) {
    if (enzyme_srets.count(i) || rroots.count(i))
      continue;

    for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i)) {
      NewAttrs = NewAttrs.addAttribute(
          F->getContext(), AttributeList::FirstArgIndex + nexti, attr);
    }
    types.push_back(F->getFunctionType()->getParamType(i));
    nexti++;
  }

  for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FunctionIndex, attr);

  FunctionType *FTy = FunctionType::get(Type::getVoidTy(F->getContext()), types,
                                        FT->isVarArg());

  // Create the new function
  auto &M = *F->getParent();
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                                    F->getName(), &M);

  ValueToValueMapTy VMap;
  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();
  Argument *sret = nullptr;
  if (sretTy) {
    sret = &*DestI;
    DestI++;
  }
  Argument *roots = nullptr;
  if (roots_AT) {
    roots = &*DestI;
    DestI++;
  }

  // To handle the deleted args, it needs to be replaced by a non-arg operand.
  // This map contains the temporary phi nodes corresponding
  std::map<size_t, PHINode *> delArgMap;
  for (Argument &I : F->args()) {
    auto i = I.getArgNo();
    if (enzyme_srets.count(i) || rroots.count(i)) {
      VMap[&I] = delArgMap[i] = PHINode::Create(I.getType(), 0);
      continue;
    }
    assert(DestI != NewF->arg_end());
    DestI->setName(I.getName()); // Copy the name over...
    VMap[&I] = &*DestI++;        // Add mapping to VMap
  }
  // Compute the readonly/nocapture/etc properties for analysis use later.
  {
    SmallPtrSet<Function *, 1> calls_todo;
    (void)DetectPointerArgOfFn(*F, calls_todo);
  }
  SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);

  SmallVector<CallInst *, 1> callers;
  for (auto U : F->users()) {
    auto CI = dyn_cast<CallInst>(U);
    assert(CI);
    assert(CI->getCalledFunction() == F);
    callers.push_back(CI);
  }

  {
    size_t curOffset = 0;
    size_t sretCount = 0;
    if (!RT->isVoidTy()) {
      for (auto &RT : Returns) {
        IRBuilder<> B(RT);
        Value *gep = ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, 0) : sret;
        Value *rval = RT->getReturnValue();
        B.CreateStore(rval, gep);

        if (roots) {
          moveSRetToFromRoots(B, rval->getType(), rval, roots_AT, roots,
                              /*rootOffset*/ 0,
                              SRetRootMovement::SRetValueToRootPointer);
        }

        auto NR = B.CreateRetVoid();
        RT->eraseFromParent();
        RT = NR;
      }
      if (roots_AT)
        curOffset = CountTrackedPointers(RT).count;
      sretCount++;
    }

    // TODO this must be re-ordered to interleave the sret/roots/etc args as
    // required.

    for (size_t i = 0, end = FT->getNumParams(); i < end; i++) {

      if (enzyme_srets.count(i)) {
        auto argFound = delArgMap.find(i);
        assert(argFound != delArgMap.end());
        auto arg = argFound->second;
        assert(arg);
        SmallVector<Instruction *, 1> uses;
        SmallVector<unsigned, 1> op;
        for (auto &U : arg->uses()) {
          auto I = cast<Instruction>(U.getUser());
          uses.push_back(I);
          op.push_back(U.getOperandNo());
        }
        IRBuilder<> EB(&NewF->getEntryBlock().front());
        auto gep =
            ST ? EB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount) : sret;
        for (size_t i = 0; i < uses.size(); i++) {
          uses[i]->setOperand(op[i], gep);
        }

        if (reroot_enzyme_srets.count(i)) {
          assert(roots_AT);
          auto cnt = CountTrackedPointers(Types[sretCount]).count;
          for (auto &RT : Returns) {
            IRBuilder<> B(RT);
            if (noroot_enzyme_srets.count(i)) {
              for (size_t i = 0; i < cnt; i++) {
                B.CreateStore(ConstantPointerNull::get(T_prjlvalue),
                              B.CreateConstInBoundsGEP2_32(roots_AT, roots, 0,
                                                           i + curOffset));
              }
            } else {
              moveSRetToFromRoots(B, Types[sretCount], gep, roots_AT, roots,
                                  curOffset,
                                  SRetRootMovement::SRetPointerToRootPointer);
            }
          }
          curOffset += cnt;
        }

        delete arg;

        sretCount++;
        continue;
      }

      if (rroots.count(i)) {
        auto attr = Attrs.getAttribute(AttributeList::FirstArgIndex + i,
                                       "enzymejl_returnRoots");
        auto attrv = attr.getValueAsString();
        assert(attrv.size());
        size_t subCount = convertRRootCountFromString(attrv);

        auto argFound = delArgMap.find(i);
        assert(argFound != delArgMap.end());
        auto arg = argFound->second;
        assert(arg);
        SmallVector<Instruction *, 1> uses;
        SmallVector<unsigned, 1> op;
        for (auto &U : arg->uses()) {
          auto I = cast<Instruction>(U.getUser());
          uses.push_back(I);
          op.push_back(U.getOperandNo());
        }
        IRBuilder<> EB(&NewF->getEntryBlock().front());

        Value *gep = nullptr;
        if (roots_AT) {
          assert(roots);
          assert(roots_AT);

          gep = roots;
          if (curOffset != 0) {
            gep = EB.CreateConstInBoundsGEP2_32(roots_AT, roots, 0, curOffset);
          }
          if (subCount != numRooting) {
            gep = EB.CreatePointerCast(
                gep, getUnqual(ArrayType::get(T_prjlvalue, subCount)));
          }
          curOffset += subCount;
          if (reret_roots.count(i))
            sretCount++;
        } else {
          assert(sret);
          gep =
              ST ? EB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount) : sret;

          if (!reret_roots.count(i)) {

            std::string s;
            llvm::raw_string_ostream ss(s);
            ss << "Illegal GC setup in which there was no roots_AT, but a new "
                  "sret ("
               << *sret << "), but no rereturned roots at index i=" << i
               << "\n";
            CustomErrorHandler(s.c_str(), wrap(gep), ErrorType::InternalError,
                               nullptr, nullptr, nullptr);
          }

          sretCount++;
        }

        for (size_t i = 0; i < uses.size(); i++) {
          uses[i]->setOperand(op[i], gep);
        }

        delete arg;
        continue;
      }
    }

    assert(curOffset == numRooting);
    assert(sretCount == Types.size());
  }

  auto &DL = F->getParent()->getDataLayout();

  // TODO fix caller side
  for (auto CI : callers) {
    auto Attrs = CI->getAttributes();
    AttributeList NewAttrs;
    IRBuilder<> B(CI);
    IRBuilder<> EB(&CI->getParent()->getParent()->getEntryBlock().front());
    SmallVector<Value *, 1> vals;
    size_t nexti = 0;
    Value *sret = nullptr;
    if (sretTy) {
      sret = EB.CreateAlloca(sretTy, 0, "stack_sret");
      vals.push_back(sret);
      NewAttrs = NewAttrs.addAttribute(
          F->getContext(), AttributeList::FirstArgIndex + nexti,
          Attribute::get(F->getContext(), Attribute::StructRet, sretTy));
      nexti++;
    }
    AllocaInst *roots = nullptr;
    if (roots_AT) {
      roots = EB.CreateAlloca(roots_AT, 0, "stack_roots_AT");
      vals.push_back(roots);
      NewAttrs = NewAttrs.addAttribute(

          F->getContext(), AttributeList::FirstArgIndex + nexti,
          "enzymejl_returnRoots", std::to_string(numRooting));
      nexti++;
    }

    for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::FunctionIndex, attr);

    SmallVector<std::tuple<Value *, Value *, Type *>> preCallReplacements;
    SmallVector<std::tuple<Value *, Value *, Type *, bool>>
        postCallReplacements;

    {
      size_t local_root_count = 0;
      size_t sretCount = 0;
      if (!RT->isVoidTy()) {
        if (roots_AT) {
          local_root_count += CountTrackedPointers(RT).count;
        }
        sretCount++;
      }

      /// TODO continue from here down for external rewrites
      for (size_t i = 0, end = CI->arg_size(); i < end; i++) {

        if (enzyme_srets.count(i)) {
          auto val = CI->getArgOperand(i);

          if (isa<UndefValue>(val) || isa<PoisonValue>(val) ||
              isa<ConstantPointerNull>(val)) {
            std::string s;
            llvm::raw_string_ostream ss(s);
            ss << "Unsupported constant argument in "
                  "FixupJuliaCallingConvention\n";
            ss << " + val: " << *val << "\n";
            ss << " + Function being rewritten: " << F->getName() << "\n";
            ss << " + CI erring: " << *CI << "\n";
            ss << " + Function containing CI: "
               << CI->getParent()->getParent()->getName() << "\n";
            if (CustomErrorHandler) {
              CustomErrorHandler(s.c_str(), wrap(CI), ErrorType::InternalError,
                                 nullptr, nullptr, nullptr);
            } else {
              EmitFailure("UnsupportedArgument", CI->getDebugLoc(), CI,
                          ss.str());
            }
          }

          Value *gep = sret;
          if (ST) {
            IRBuilder<> GEPB(cast<Instruction>(sret)->getNextNode());
            gep = GEPB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount);
          }

          bool handled = false;
          if (auto AI = dyn_cast<AllocaInst>(getBaseObject(val, false))) {
            if (AI->getAllocatedType() == Types[sretCount] ||
                (isOpaque(AI->getType()) &&
                 DL.getTypeSizeInBits(AI->getAllocatedType()) ==
                     DL.getTypeSizeInBits(Types[sretCount]))) {
              AI->replaceAllUsesWith(gep);
              AI->eraseFromParent();
              handled = true;
            }
          }

          if (!handled) {
            assert(!isa<UndefValue>(val));
            assert(!isa<PoisonValue>(val));
            assert(!isa<ConstantPointerNull>(val));

            // On Julia 1.12+, the sret does not actually contain the jlvaluet
            // (and it should not). However, if the sret does not contain a
            // return roots (per tracked pointers), we do still need to perform
            // the store.
            bool should_sret = sret_jlvalue;
            if (!should_sret) {
              CountTrackedPointers tracked(Types[sretCount]);
              if (tracked.count && tracked.all)
                should_sret = true;
            }

            // Don't bother to copy back in if the original function doesn't
            // store anything.
            bool copyBack = !isReadOnlyNoCapture(F, i);
            if (copyBack) {
              postCallReplacements.emplace_back(val, gep, Types[sretCount],
                                                should_sret);
            }
            // Only copy in the inital value if the function reads, or we are
            // going to copy back and the function doesn't store all bytes.
            if (!isWriteOnly(CI, i) ||
                (copyBack &&
                 !isGuaranteedToFullyWrite(F, i, Types[sretCount]))) {
              preCallReplacements.emplace_back(val, gep, Types[sretCount]);
            }
          }

          if (roots_AT && reroot_enzyme_srets.count(i)) {
            local_root_count += CountTrackedPointers(Types[sretCount]).count;
          }

          sretCount++;
          continue;
        }

        if (rroots.count(i)) {
          auto val = CI->getArgOperand(i);
          if (isa<UndefValue>(val) || isa<PoisonValue>(val) ||
              isa<ConstantPointerNull>(val)) {
            std::string s;
            llvm::raw_string_ostream ss(s);
            ss << "Unsupported constant argument in "
                  "FixupJuliaCallingConvention\n";
            ss << " + val: " << *val << "\n";
            ss << " + Function being rewritten: " << F->getName() << "\n";
            ss << " + CI erring: " << *CI << "\n";
            ss << " + Function containing CI: "
               << CI->getParent()->getParent()->getName() << "\n";
            if (CustomErrorHandler) {
              CustomErrorHandler(s.c_str(), wrap(CI), ErrorType::InternalError,
                                 nullptr, nullptr, nullptr);
            } else {
              EmitFailure("UnsupportedArgument", CI->getDebugLoc(), CI,
                          ss.str());
            }
          }

          auto attr = Attrs.getAttribute(AttributeList::FirstArgIndex + i,
                                         "enzymejl_returnRoots");
          auto attrv = attr.getValueAsString();
          assert(attrv.size());
          size_t subCount = convertRRootCountFromString(attrv);

          Value *gep = nullptr;

          if (roots_AT) {
            assert(roots);
            IRBuilder<> GEPB(cast<Instruction>(roots)->getNextNode());
            gep = roots;
            if (local_root_count != 0) {
              gep = GEPB.CreateConstInBoundsGEP2_32(roots_AT, roots, 0,
                                                    local_root_count);
            }

            if (subCount != numRooting) {
              gep = GEPB.CreatePointerCast(
                  gep, getUnqual(ArrayType::get(T_prjlvalue, subCount)));
            }
            local_root_count += subCount;
            if (reret_roots.count(i))
              sretCount++;
          } else {
            assert(reret_roots.count(i));
            assert(sret);
            IRBuilder<> GEPB(cast<Instruction>(sret)->getNextNode());
            gep = sret;
            if (ST) {
              gep = GEPB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount);
            }
            sretCount++;
          }

          bool handled = false;
          if (auto AI = dyn_cast<AllocaInst>(getBaseObject(val, false))) {
            if (AI->getAllocatedType() ==
                ArrayType::get(T_prjlvalue, subCount)) {
              AI->replaceAllUsesWith(gep);
              AI->eraseFromParent();
              handled = true;
            }
          }

          if (!handled) {
            assert(!isa<UndefValue>(val));
            assert(!isa<PoisonValue>(val));
            assert(!isa<ConstantPointerNull>(val));
            // TODO consider doing pre-emptive pre zero of the section?
            preCallReplacements.emplace_back(
                val, gep, ArrayType::get(T_prjlvalue, subCount));
            postCallReplacements.emplace_back(
                val, gep, ArrayType::get(T_prjlvalue, subCount), true);
          }
          continue;
        }

        for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i))
          NewAttrs = NewAttrs.addAttribute(
              F->getContext(), AttributeList::FirstArgIndex + nexti, attr);
        vals.push_back(CI->getArgOperand(i));
        nexti++;
      }

      assert(sretCount == Types.size());
      assert(local_root_count == numRooting);
    }

    // Because we will += into the corresponding derivative sret, we need to
    // pass in the values that were actually there before the call
    // TODO we can optimize this further and avoid the copy in the primal and/or
    // forward mode as the copy is _only_ needed for the adjoint.
    for (auto &&[val, gep, ty] : preCallReplacements) {
      copyNonJLValueInto(B, ty, ty, gep, {}, ty, val, {}, /*shouldZero*/ true);
    }

    // Actually perform the call, copying over relevant information.
    SmallVector<OperandBundleDef, 1> Bundles;
    for (unsigned I = 0, E = CI->getNumOperandBundles(); I != E; ++I)
      Bundles.emplace_back(CI->getOperandBundleAt(I));

    if (!NewF->getFunctionType()->isVarArg() &&
        NewF->getFunctionType()->getNumParams() != vals.size()) {
      llvm::errs() << "NewF: " << *NewF << "\n";
      for (size_t i = 0; i < vals.size(); i++) {
        llvm::errs() << " Args[" << i << "] = " << *vals[i] << "\n";
      }
    }
    auto NC = B.CreateCall(NewF, vals, Bundles);
    NC->setAttributes(NewAttrs);

    SmallVector<std::pair<unsigned, MDNode *>, 4> TheMDs;
    CI->getAllMetadataOtherThanDebugLoc(TheMDs);
    SmallVector<unsigned, 1> toCopy;
    for (auto pair : TheMDs)
      if (pair.first != LLVMContext::MD_range) {
        toCopy.push_back(pair.first);
      }
    if (!toCopy.empty())
      NC->copyMetadata(*CI, toCopy);
    NC->setDebugLoc(CI->getDebugLoc());

    if (!RT->isVoidTy()) {
      auto gep = ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, 0) : sret;
      auto ld = B.CreateLoad(RT, gep);
      if (auto MD = CI->getMetadata(LLVMContext::MD_range))
        ld->setMetadata(LLVMContext::MD_range, MD);
      ld->takeName(CI);
      Value *replacement = ld;

      // We don't need to override the jlvalue_t's with the rooted versions here
      // since we already stored the full value into the sret above.
      // if (fromRoots) {
      //  replacement = moveSRetToFromRoots(B, replacement->getType(),
      //  replacement, root_AT, root, /*rootOffset*/0,
      //  SRetRootMovement::RootPointerToSRetValue);
      //}

      CI->replaceAllUsesWith(replacement);
    }

    for (auto &&[val, gep, ty, jlvalue] : postCallReplacements) {
      if (jlvalue) {
        auto ld = B.CreateLoad(ty, gep);
        auto SI = B.CreateStore(ld, val);
        if (val->getType()->getPointerAddressSpace() == 10)
          PostCacheStore(SI, B);
      } else {
        copyNonJLValueInto(B, ty, ty, val, {}, ty, gep, {},
                           /*shouldZero*/ false);
      }
    }

    NC->setCallingConv(CI->getCallingConv());
    CI->eraseFromParent();
  }
  NewF->setAttributes(NewAttrs);
  SmallVector<std::pair<unsigned, MDNode *>, 1> MD;
  F->getAllMetadata(MD);
  for (auto pair : MD)
    if (pair.first != LLVMContext::MD_dbg)
      NewF->addMetadata(pair.first, *pair.second);
  NewF->takeName(F);
  NewF->setCallingConv(F->getCallingConv());
  F->eraseFromParent();
}

#include "llvm/Passes/PassBuilder.h"

#include <string>

using namespace llvm;

void EnzymeFixupBatchedJuliaCallingConvention(Function *F) {
  if (F->empty())
    return;
  auto RT = F->getReturnType();
  auto FT = F->getFunctionType();
  auto Attrs = F->getAttributes();

  AttributeList NewAttrs;
  SmallVector<Type *, 1> types;
  SmallSet<size_t, 1> changed;
  for (auto pair : llvm::enumerate(FT->params())) {
    auto T = pair.value();
    auto i = pair.index();
    bool sretv = false;
    StringRef kind;
    StringRef value;
    for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i)) {
      if (attr.isStringAttribute() &&
          attr.getKindAsString() == "enzyme_sret_v") {
        sretv = true;
        kind = "enzyme_sret";
        value = attr.getValueAsString();
      } else if (attr.isStringAttribute() &&
                 attr.getKindAsString() == "enzymejl_rooted_typ_v") {
        sretv = true;
        kind = "enzymejl_rooted_typ";
        value = attr.getValueAsString();
      } else if (attr.isStringAttribute() &&
                 attr.getKindAsString() == "enzymejl_returnRoots_v") {
        sretv = true;
        kind = "enzymejl_returnRoots";
        value = attr.getValueAsString();
      } else {
        NewAttrs = NewAttrs.addAttribute(
            F->getContext(), AttributeList::FirstArgIndex + types.size(), attr);
      }
    }
    if (auto AT = dyn_cast<ArrayType>(T)) {
      if (auto PT = dyn_cast<PointerType>(AT->getElementType())) {
        auto AS = PT->getAddressSpace();
        if (AS == 11 || AS == 12 || AS == 13 || sretv) {
          for (unsigned i = 0; i < AT->getNumElements(); i++) {
            if (sretv) {
              NewAttrs = NewAttrs.addAttribute(
                  F->getContext(), AttributeList::FirstArgIndex + types.size(),
                  Attribute::get(F->getContext(), kind, value));
            }
            types.push_back(PT);
          }
          changed.insert(i);
          continue;
        }
      }
    }
    assert(!sretv);
    types.push_back(T);
  }
  if (changed.size() == 0)
    return;

  for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FunctionIndex, attr);

  for (auto attr : Attrs.getAttributes(AttributeList::ReturnIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::ReturnIndex, attr);

  FunctionType *FTy =
      FunctionType::get(FT->getReturnType(), types, FT->isVarArg());

  // Create the new function
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                                    F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();

  // To handle the deleted args, it needs to be replaced by a non-arg operand.
  // This map contains the temporary phi nodes corresponding
  SmallVector<Instruction *, 1> toInsert;
  for (Argument &I : F->args()) {
    auto T = I.getType();
    if (auto AT = dyn_cast<ArrayType>(T)) {
      if (changed.count(I.getArgNo())) {
        Value *V = UndefValue::get(T);
        for (unsigned i = 0; i < AT->getNumElements(); i++) {
          DestI->setName(I.getName() + "." +
                         std::to_string(i)); // Copy the name over...
          unsigned idx[1] = {i};
          auto IV = InsertValueInst::Create(V, (llvm::Value *)&*DestI++, idx);
          toInsert.push_back(IV);
          V = IV;
        }
        VMap[&I] = V;
        continue;
      }
    }
    DestI->setName(I.getName()); // Copy the name over...
    VMap[&I] = &*DestI++;        // Add mapping to VMap
  }

  SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);

  {
    IRBuilder<> EB(&*NewF->getEntryBlock().begin());
    for (auto I : toInsert)
      EB.Insert(I);
  }

  SmallVector<CallInst *, 1> callers;
  for (auto U : F->users()) {
    auto CI = dyn_cast<CallInst>(U);
    assert(CI);
    assert(CI->getCalledFunction() == F);
    callers.push_back(CI);
  }

  for (auto CI : callers) {
    auto Attrs = CI->getAttributes();
    AttributeList NewAttrs;
    IRBuilder<> B(CI);

    for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::FunctionIndex, attr);

    for (auto attr : Attrs.getAttributes(AttributeList::ReturnIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::ReturnIndex, attr);

    SmallVector<Value *, 1> vals;
    for (size_t j = 0, end = CI->arg_size(); j < end; j++) {

      auto T = CI->getArgOperand(j)->getType();
      if (auto AT = dyn_cast<ArrayType>(T)) {
        if (isa<PointerType>(AT->getElementType())) {
          if (changed.count(j)) {
            bool sretv = false;
            std::string kind;
            StringRef value;
            for (auto attr :
                 Attrs.getAttributes(AttributeList::FirstArgIndex + j)) {
              if (attr.isStringAttribute() &&
                  attr.getKindAsString() == "enzyme_sret_v") {
                sretv = true;
                kind = "enzyme_sret";
                value = attr.getValueAsString();
              } else if (attr.isStringAttribute() &&
                         attr.getKindAsString() == "enzymejl_returnRoots_v") {
                sretv = true;
                kind = "enzymejl_returnRoots";
                value = attr.getValueAsString();
              } else if (attr.isStringAttribute() &&
                         attr.getKindAsString() == "enzymejl_rooted_typ_v") {
                sretv = true;
                kind = "enzymejl_rooted_typ_v";
                value = attr.getValueAsString();
              }
            }
            for (unsigned i = 0; i < AT->getNumElements(); i++) {
              if (sretv)
                NewAttrs = NewAttrs.addAttribute(
                    F->getContext(), AttributeList::FirstArgIndex + vals.size(),
                    Attribute::get(F->getContext(), kind, value));
              vals.push_back(
                  GradientUtils::extractMeta(B, CI->getArgOperand(j), i));
            }
            continue;
          }
        }
      }

      for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + j)) {
        if (attr.isStringAttribute() &&
            attr.getKindAsString() == "enzyme_sret_v") {
          NewAttrs = NewAttrs.addAttribute(
              F->getContext(), AttributeList::FirstArgIndex + vals.size(),
              Attribute::get(F->getContext(), "enzyme_sret",
                             attr.getValueAsString()));
        } else if (attr.isStringAttribute() &&
                   attr.getKindAsString() == "enzymejl_returnRoots_v") {
          NewAttrs = NewAttrs.addAttribute(
              F->getContext(), AttributeList::FirstArgIndex + vals.size(),
              Attribute::get(F->getContext(), "enzymejl_returnRoots",
                             attr.getValueAsString()));
        } else if (attr.isStringAttribute() &&
                   attr.getKindAsString() == "enzymejl_rooted_typ_v") {
          NewAttrs = NewAttrs.addAttribute(
              F->getContext(), AttributeList::FirstArgIndex + vals.size(),
              Attribute::get(F->getContext(), "enzymejl_rooted_typ",
                             attr.getValueAsString()));
        } else {
          NewAttrs = NewAttrs.addAttribute(
              F->getContext(), AttributeList::FirstArgIndex + vals.size(),
              attr);
        }
      }

      vals.push_back(CI->getArgOperand(j));
    }

    SmallVector<OperandBundleDef, 1> Bundles;
    for (unsigned I = 0, E = CI->getNumOperandBundles(); I != E; ++I)
      Bundles.emplace_back(CI->getOperandBundleAt(I));
    auto NC = B.CreateCall(NewF, vals, Bundles);
    NC->setAttributes(NewAttrs);

    SmallVector<std::pair<unsigned, MDNode *>, 4> TheMDs;
    CI->getAllMetadataOtherThanDebugLoc(TheMDs);
    SmallVector<unsigned, 1> toCopy;
    for (auto pair : TheMDs)
      toCopy.push_back(pair.first);
    if (!toCopy.empty())
      NC->copyMetadata(*CI, toCopy);
    NC->setDebugLoc(CI->getDebugLoc());

    if (!RT->isVoidTy()) {
      NC->takeName(CI);
      CI->replaceAllUsesWith(NC);
    }

    NC->setCallingConv(CI->getCallingConv());
    CI->eraseFromParent();
  }
  NewF->setAttributes(NewAttrs);
  SmallVector<std::pair<unsigned, MDNode *>, 1> MD;
  F->getAllMetadata(MD);
  for (auto pair : MD)
    if (pair.first != LLVMContext::MD_dbg)
      NewF->addMetadata(pair.first, *pair.second);
  NewF->takeName(F);
  NewF->setCallingConv(F->getCallingConv());
  F->eraseFromParent();
}

class FixupJuliaCallingConventionNewPM
    : public PassInfoMixin<FixupJuliaCallingConventionNewPM> {
  bool sret_jlvalue;

public:
  FixupJuliaCallingConventionNewPM(bool sret_jlvalue)
      : sret_jlvalue(sret_jlvalue) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    bool changed = false;
    SmallVector<llvm::Function *, 16> Functions;
    for (auto &F : M) {
      if (F.empty())
        continue;
      Functions.push_back(&F);
    }
    for (auto *F : Functions) {
      EnzymeFixupJuliaCallingConvention(F, sret_jlvalue);
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

class FixupBatchedJuliaCallingConventionNewPM
    : public PassInfoMixin<FixupBatchedJuliaCallingConventionNewPM> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    bool changed = false;
    SmallVector<llvm::Function *, 16> Functions;
    for (auto &F : M) {
      if (F.empty())
        continue;
      Functions.push_back(&F);
    }
    for (auto *F : Functions) {
      EnzymeFixupBatchedJuliaCallingConvention(F);
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Expose New PM pass for registration
bool registerFixupJuliaPass(StringRef Name, ModulePassManager &MPM) {
  if (Name == "enzyme-fixup-julia") {
    MPM.addPass(FixupJuliaCallingConventionNewPM(false));
    return true;
  }
  if (Name == "enzyme-fixup-julia-sret") {
    MPM.addPass(FixupJuliaCallingConventionNewPM(true));
    return true;
  }
  if (Name == "enzyme-fixup-batched-julia") {
    MPM.addPass(FixupBatchedJuliaCallingConventionNewPM());
    return true;
  }
  return false;
}
