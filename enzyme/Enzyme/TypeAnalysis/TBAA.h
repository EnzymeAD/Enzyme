//===- TBAA.h - Helpers for llvm::Type-based alias analysis   ------------===//
//
//                   Enzyme Project and The LLVM Project
// First section modified from: TypeBasedAliasAnalysis.cpp in LLVM
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
// This file contains the implementation of several utilities for understanding
// TBAA metadata and converting that metadata into corresponding TypeAnalysis
// representations.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TYPE_ANALYSIS_TBAA_H
#define ENZYME_TYPE_ANALYSIS_TBAA_H 1

#include "BaseType.h"
#include "ConcreteType.h"
#include "TypeTree.h"
#include <algorithm>

/// isNewFormatTypeNode - Return true iff the given type node is in the new
/// size-aware format.
static bool isNewFormatTypeNode(const llvm::MDNode *N) {
  if (N->getNumOperands() < 3)
    return false;
  // In the old format the first operand is a string.
  if (!llvm::isa<llvm::MDNode>(N->getOperand(0)))
    return false;
  return true;
}

/// This is a simple wrapper around an llvm::MDNode which provides a
/// higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
template <typename MDNodeTy> class TBAANodeImpl {
  MDNodeTy *Node = nullptr;

public:
  TBAANodeImpl() = default;
  explicit TBAANodeImpl(MDNodeTy *N) : Node(N) {}

  /// getNode - Get the llvm::MDNode for this TBAANode.
  MDNodeTy *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped type node is in the new
  /// size-aware format.
  bool isNewFormat() const { return isNewFormatTypeNode(Node); }

  /// getParent - Get this TBAANode's Alias tree parent.
  TBAANodeImpl<MDNodeTy> getParent() const {
    if (isNewFormat())
      return TBAANodeImpl(llvm::cast<MDNodeTy>(Node->getOperand(0)));

    if (Node->getNumOperands() < 2)
      return TBAANodeImpl<MDNodeTy>();
    MDNodeTy *P = llvm::dyn_cast_or_null<MDNodeTy>(Node->getOperand(1));
    if (!P)
      return TBAANodeImpl<MDNodeTy>();
    // Ok, this node has a valid parent. Return it.
    return TBAANodeImpl<MDNodeTy>(P);
  }

  /// Test if this TBAANode represents a type for objects which are
  /// not modified (by any means) in the context where this
  /// AliasAnalysis is relevant.
  bool isTypeImmutable() const {
    if (Node->getNumOperands() < 3)
      return false;
    llvm::ConstantInt *CI =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(Node->getOperand(2));
    if (!CI)
      return false;
    return CI->getValue()[0];
  }
};

/// \name Specializations of \c TBAANodeImpl for const and non const qualified
/// \c MDNode.
/// @{
using TBAANode = TBAANodeImpl<const llvm::MDNode>;
using MutableTBAANode = TBAANodeImpl<llvm::MDNode>;
/// @}

/// This is a simple wrapper around an llvm::MDNode which provides a
/// higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
template <typename MDNodeTy> class TBAAStructTagNodeImpl {
  /// This node should be created with createTBAAAccessTag().
  MDNodeTy *Node;

public:
  explicit TBAAStructTagNodeImpl(MDNodeTy *N) : Node(N) {}

  /// Get the llvm::MDNode for this TBAAStructTagNode.
  MDNodeTy *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped access tag is in the new
  /// size-aware format.
  bool isNewFormat() const {
    if (Node->getNumOperands() < 4)
      return false;
    if (MDNodeTy *AccessType = getAccessType())
      if (!TBAANodeImpl<MDNodeTy>(AccessType).isNewFormat())
        return false;
    return true;
  }

  MDNodeTy *getBaseType() const {
    return llvm::dyn_cast_or_null<llvm::MDNode>(Node->getOperand(0));
  }

  MDNodeTy *getAccessType() const {
    return llvm::dyn_cast_or_null<llvm::MDNode>(Node->getOperand(1));
  }

  uint64_t getOffset() const {
    return llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(2))
        ->getZExtValue();
  }

  uint64_t getSize() const {
    if (!isNewFormat())
      return UINT64_MAX;
    return llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(3))
        ->getZExtValue();
  }

  /// Test if this TBAAStructTagNode represents a type for objects
  /// which are not modified (by any means) in the context where this
  /// AliasAnalysis is relevant.
  bool isTypeImmutable() const {
    unsigned OpNo = isNewFormat() ? 4 : 3;
    if (Node->getNumOperands() < OpNo + 1)
      return false;
    llvm::ConstantInt *CI =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(Node->getOperand(OpNo));
    if (!CI)
      return false;
    return CI->getValue()[0];
  }
};

/// \name Specializations of \c TBAAStructTagNodeImpl for const and non const
/// qualified \c MDNods.
/// @{
using TBAAStructTagNode = TBAAStructTagNodeImpl<const llvm::MDNode>;
using MutableTBAAStructTagNode = TBAAStructTagNodeImpl<llvm::MDNode>;
/// @}

/// This is a simple wrapper around an llvm::MDNode which provides a
/// higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
class TBAAStructTypeNode {
  /// This node should be created with createTBAATypeNode().
  const llvm::MDNode *Node = nullptr;

public:
  TBAAStructTypeNode() = default;
  explicit TBAAStructTypeNode(const llvm::MDNode *N) : Node(N) {}

  /// Get the llvm::MDNode for this TBAAStructTypeNode.
  const llvm::MDNode *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped type node is in the new
  /// size-aware format.
  bool isNewFormat() const { return isNewFormatTypeNode(Node); }

  bool operator==(const TBAAStructTypeNode &Other) const {
    return getNode() == Other.getNode();
  }

  /// getId - Return type identifier.
  llvm::Metadata *getId() const {
    return Node->getOperand(isNewFormat() ? 2 : 0);
  }

  unsigned getNumFields() const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    return (getNode()->getNumOperands() - FirstFieldOpNo) / NumOpsPerField;
  }

  uint64_t getFieldOffset(unsigned FieldIndex) const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    unsigned OpIndex = FirstFieldOpNo + FieldIndex * NumOpsPerField;

    uint64_t Cur =
        llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(OpIndex + 1))
            ->getZExtValue();
    return Cur;
  }

  TBAAStructTypeNode getFieldType(unsigned FieldIndex) const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    unsigned OpIndex = FirstFieldOpNo + FieldIndex * NumOpsPerField;
    auto *TypeNode = llvm::cast<llvm::MDNode>(getNode()->getOperand(OpIndex));
    return TBAAStructTypeNode(TypeNode);
  }

  /// Get this TBAAStructTypeNode's field in the type DAG with
  /// given offset. Update the offset to be relative to the field type.
  TBAAStructTypeNode getField(uint64_t &Offset) const {
    bool NewFormat = isNewFormat();
    if (NewFormat) {
      // New-format root and scalar type nodes have no fields.
      if (Node->getNumOperands() < 6)
        return TBAAStructTypeNode();
    } else {
      // Parent can be omitted for the root node.
      if (Node->getNumOperands() < 2)
        return TBAAStructTypeNode();

      // Fast path for a scalar type node and a struct type node with a single
      // field.
      if (Node->getNumOperands() <= 3) {
        uint64_t Cur =
            Node->getNumOperands() == 2
                ? 0
                : llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(2))
                      ->getZExtValue();
        Offset -= Cur;
        llvm::MDNode *P =
            llvm::dyn_cast_or_null<llvm::MDNode>(Node->getOperand(1));
        if (!P)
          return TBAAStructTypeNode();
        return TBAAStructTypeNode(P);
      }
    }

    // Assume the offsets are in order. We return the previous field if
    // the current offset is bigger than the given offset.
    unsigned FirstFieldOpNo = NewFormat ? 3 : 1;
    unsigned NumOpsPerField = NewFormat ? 3 : 2;
    unsigned TheIdx = 0;
    for (unsigned Idx = FirstFieldOpNo; Idx < Node->getNumOperands();
         Idx += NumOpsPerField) {
      uint64_t Cur =
          llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(Idx + 1))
              ->getZExtValue();
      if (Cur > Offset) {
        assert(Idx >= FirstFieldOpNo + NumOpsPerField &&
               "TBAAStructTypeNode::getField should have an offset match!");
        TheIdx = Idx - NumOpsPerField;
        break;
      }
    }
    // Move along the last field.
    if (TheIdx == 0)
      TheIdx = Node->getNumOperands() - NumOpsPerField;
    uint64_t Cur =
        llvm::mdconst::extract<llvm::ConstantInt>(Node->getOperand(TheIdx + 1))
            ->getZExtValue();
    Offset -= Cur;
    llvm::MDNode *P =
        llvm::dyn_cast_or_null<llvm::MDNode>(Node->getOperand(TheIdx));
    if (!P)
      return TBAAStructTypeNode();
    return TBAAStructTypeNode(P);
  }
};

/// Check the first operand of the tbaa tag node, if it is a llvm::MDNode, we
/// treat it as struct-path aware TBAA format, otherwise, we treat it as scalar
/// TBAA format.
static inline bool isStructPathTBAA(const llvm::MDNode *MD) {
  // Anonymous TBAA root starts with a llvm::MDNode and dragonegg uses it as
  // a TBAA tag.
  return llvm::isa<llvm::MDNode>(MD->getOperand(0)) &&
         MD->getNumOperands() >= 3;
}

static inline const llvm::MDNode *
createAccessTag(const llvm::MDNode *AccessType) {
  // If there is no access type or the access type is the root node, then
  // we don't have any useful access tag to return.
  if (!AccessType || AccessType->getNumOperands() < 2)
    return nullptr;

  llvm::Type *Int64 = llvm::IntegerType::get(AccessType->getContext(), 64);
  auto *OffsetNode =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int64, 0));

  if (TBAAStructTypeNode(AccessType).isNewFormat()) {
    // TODO: Take access ranges into account when matching access tags and
    // fix this code to generate actual access sizes for generic tags.
    uint64_t AccessSize = UINT64_MAX;
    auto *SizeNode = llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(Int64, AccessSize));
    llvm::Metadata *Ops[] = {const_cast<llvm::MDNode *>(AccessType),
                             const_cast<llvm::MDNode *>(AccessType), OffsetNode,
                             SizeNode};
    return llvm::MDNode::get(AccessType->getContext(), Ops);
  }

  llvm::Metadata *Ops[] = {const_cast<llvm::MDNode *>(AccessType),
                           const_cast<llvm::MDNode *>(AccessType), OffsetNode};
  return llvm::MDNode::get(AccessType->getContext(), Ops);
}

// Modified from llvm::MDNode::isTBAAVtableAccess()

static inline std::string
getAccessNameTBAA(const llvm::MDNode *M,
                  const std::set<std::string> &legalnames) {
  if (!isStructPathTBAA(M)) {
    if (M->getNumOperands() < 1)
      return "";
    if (const llvm::MDString *Tag1 =
            llvm::dyn_cast<llvm::MDString>(M->getOperand(0))) {
      return Tag1->getString().str();
    }
    return "";
  }

  // For struct-path aware TBAA, we use the access type of the tag.
  // llvm::errs() << "M: " << *M << "\n";
  TBAAStructTagNode Tag(M);
  // llvm::errs() << "AT: " << *Tag.getAccessType() << "\n";
  TBAAStructTypeNode AccessType(Tag.getAccessType());

  // llvm::errs() << "numfields: " << AccessType.getNumFields() << "\n";
  while (AccessType.getNumFields() > 0) {

    if (auto *Id = llvm::dyn_cast<llvm::MDString>(AccessType.getId())) {
      // llvm::errs() << "cur access type: " << Id->getString() << "\n";
      if (legalnames.count(Id->getString().str())) {
        return Id->getString().str();
      }
    }

    AccessType = AccessType.getFieldType(0);
    // llvm::errs() << "numfields: " << AccessType.getNumFields() << "\n";
  }

  if (auto *Id = llvm::dyn_cast<llvm::MDString>(AccessType.getId())) {
    // llvm::errs() << "access type: " << Id->getString() << "\n";
    return Id->getString().str();
  }
  return "";
}

static inline std::string
getAccessNameTBAA(llvm::Instruction *Inst,
                  const std::set<std::string> &legalnames) {
  if (const llvm::MDNode *M =
          Inst->getMetadata(llvm::LLVMContext::MD_tbaa_struct)) {
    for (unsigned i = 2; i < M->getNumOperands(); i += 3) {
      if (const llvm::MDNode *M2 =
              llvm::dyn_cast<llvm::MDNode>(M->getOperand(i))) {
        auto res = getAccessNameTBAA(M2, legalnames);
        if (res != "")
          return res;
      }
    }
  }
  if (const llvm::MDNode *M = Inst->getMetadata(llvm::LLVMContext::MD_tbaa)) {
    return getAccessNameTBAA(M, legalnames);
  }
  return "";
}

//! The following is not taken from LLVM

extern "C" {
/// Flag to print llvm::Type Analysis results as they are derived
extern llvm::cl::opt<bool> EnzymePrintType;
}

/// Return whether TypeName is a Clang _BitInt(N) scalar TBAA identifier.
static inline bool isBitIntTBAATypeName(llvm::StringRef TypeName) {
  if (!startsWith(TypeName, "_BitInt(") || !endsWith(TypeName, ")"))
    return false;
  llvm::StringRef Width =
      TypeName.drop_front(strlen("_BitInt(")).drop_back(strlen(")"));
  return !Width.empty() && llvm::all_of(Width, [](char Digit) {
    return Digit >= '0' && Digit <= '9';
  });
}

/// Return whether TypeName is a Clang pointer TBAA scalar identifier such as
/// "p1 int" or "any p2 pointer".
static inline bool isPointerTBAATypeName(llvm::StringRef TypeName) {
  if (startsWith(TypeName, "any p") && endsWith(TypeName, " pointer")) {
    llvm::StringRef Depth =
        TypeName.drop_front(strlen("any p")).drop_back(strlen(" pointer"));
    return !Depth.empty() && llvm::all_of(Depth, [](char Digit) {
      return Digit >= '0' && Digit <= '9';
    });
  }
  if (startsWith(TypeName, "p")) {
    size_t SpaceIndex = TypeName.find(' ');
    if (SpaceIndex != llvm::StringRef::npos && SpaceIndex > 1 &&
        SpaceIndex + 1 < TypeName.size()) {
      llvm::StringRef Depth = TypeName.slice(1, SpaceIndex);
      return llvm::all_of(
          Depth, [](char Digit) { return Digit >= '0' && Digit <= '9'; });
    }
  }
  return false;
}

/// Derive the ConcreteType corresponding to the string TypeName
/// The llvm::Instruction I denotes the context in which this was found
static inline ConcreteType
getTypeFromTBAAString(std::string TypeName, llvm::Instruction &I,
                      std::shared_ptr<llvm::ModuleSlotTracker> MST) {
  ConcreteType Result(BaseType::Unknown);
  if (TypeName == "long long" || TypeName == "long" || TypeName == "int" ||
      TypeName == "short" || TypeName == "bool" || TypeName == "wchar_t" ||
      TypeName == "char8_t" || TypeName == "char16_t" ||
      TypeName == "char32_t" || TypeName == "__int128" ||
      isBitIntTBAATypeName(TypeName) || TypeName == "jtbaa_arraysize" ||
      TypeName == "jtbaa_arraylen") {
    Result = ConcreteType(BaseType::Integer);
  } else if (TypeName == "any pointer" || TypeName == "vtable pointer" ||
             TypeName == "jtbaa_tag" || isPointerTBAATypeName(TypeName)) {
    Result = ConcreteType(BaseType::Pointer);
  } else if (TypeName == "half" || TypeName == "_Float16") {
    Result = llvm::Type::getHalfTy(I.getContext());
  } else if (TypeName == "__bf16") {
    Result = llvm::Type::getBFloatTy(I.getContext());
  } else if (TypeName == "float") {
    Result = llvm::Type::getFloatTy(I.getContext());
  } else if (TypeName == "double") {
    Result = llvm::Type::getDoubleTy(I.getContext());
  } else if (TypeName == "__float128") {
    Result = llvm::Type::getFP128Ty(I.getContext());
  }
  if (Result.isKnown() && EnzymePrintType) {
    llvm::errs() << "known tbaa ";
    if (MST)
      I.print(llvm::errs(), *MST);
    else
      llvm::errs() << I;
    llvm::errs() << ' ' << TypeName << '\n';
  }
  return Result;
}

/// Combine TBAA field TypeTrees while dropping entries whose outer byte spans
/// overlap with incompatible types, such as union alternatives or bitfields.
static inline TypeTree
mergeNonConflictingTBAATrees(llvm::ArrayRef<TypeTree> Trees,
                             const llvm::DataLayout &DL) {
  struct TBAASpanEntry {
    std::vector<int> Offsets;
    ConcreteType Type;
    uint64_t ByteStart;
    uint64_t ByteEnd;
    uint64_t ElementBytes;
    bool Conflicted = false;
  };

  uint64_t PointerBytes = (DL.getPointerSizeInBits() + 7) / 8;
  if (PointerBytes == 0)
    PointerBytes = 1;

  llvm::SmallVector<TBAASpanEntry, 8> Entries;
  for (const TypeTree &Tree : Trees) {
    for (const auto &MappingPair : Tree.getMapping()) {
      if (MappingPair.first.empty())
        continue;
      uint64_t ElementBytes = 1;
      if (MappingPair.first.size() > 1 ||
          MappingPair.second == BaseType::Pointer) {
        ElementBytes = PointerBytes;
      } else if (llvm::Type *FloatType = MappingPair.second.isFloat()) {
        ElementBytes = (DL.getTypeSizeInBits(FloatType) + 7) / 8;
      }
      if (ElementBytes == 0)
        ElementBytes = 1;

      uint64_t ByteStart = 0;
      uint64_t ByteEnd = UINT64_MAX;
      if (MappingPair.first[0] != -1) {
        ByteStart = static_cast<uint64_t>(MappingPair.first[0]);
        ByteEnd = (UINT64_MAX - ByteStart < ElementBytes)
                      ? UINT64_MAX
                      : ByteStart + ElementBytes;
      }
      Entries.push_back({MappingPair.first, MappingPair.second, ByteStart,
                         ByteEnd, ElementBytes, false});
    }
  }

  std::stable_sort(Entries.begin(), Entries.end(),
                   [](const TBAASpanEntry &Left, const TBAASpanEntry &Right) {
                     return Left.ByteStart < Right.ByteStart;
                   });

  auto entriesConflict = [](const TBAASpanEntry &First,
                            const TBAASpanEntry &Second) {
    if (First.Offsets[0] != -1 && Second.Offsets[0] != -1) {
      if (First.ByteStart != Second.ByteStart)
        return true;
    } else {
      if (First.ElementBytes != Second.ElementBytes)
        return true;
      if (First.Offsets[0] != -1 && First.ByteStart % Second.ElementBytes != 0)
        return true;
      if (Second.Offsets[0] != -1 && Second.ByteStart % First.ElementBytes != 0)
        return true;
    }
    const TBAASpanEntry &Shorter =
        First.Offsets.size() <= Second.Offsets.size() ? First : Second;
    const TBAASpanEntry &Longer =
        First.Offsets.size() <= Second.Offsets.size() ? Second : First;
    TypeTree Probe;
    bool Legal = true;
    Probe.checkedOrIn(Shorter.Offsets, Shorter.Type, /*PointerIntSame=*/false,
                      Legal);
    if (Legal)
      Probe.checkedOrIn(Longer.Offsets, Longer.Type, /*PointerIntSame=*/false,
                        Legal);
    return !Legal;
  };

  for (size_t i = 0, End = Entries.size(); i < End; ++i) {
    for (size_t j = i + 1; j < End; ++j) {
      if (Entries[j].ByteStart >= Entries[i].ByteEnd)
        break;
      if (entriesConflict(Entries[i], Entries[j])) {
        Entries[i].Conflicted = true;
        Entries[j].Conflicted = true;
      }
    }
  }

  TypeTree Result(BaseType::Pointer);
  for (const TBAASpanEntry &Entry : Entries) {
    if (!Entry.Conflicted)
      Result.orIn(Entry.Offsets, Entry.Type);
  }
  return Result;
}

/// Given a TBAA access node return the corresponding TypeTree
/// This includes recursively parsing the access nodes, with
/// corresponding offsets in the result
static inline TypeTree parseTBAA(TBAAStructTypeNode AccessType,
                                 llvm::Instruction &I,
                                 const llvm::DataLayout &DL,
                                 std::shared_ptr<llvm::ModuleSlotTracker> MST) {

  if (auto *Id = llvm::dyn_cast<llvm::MDString>(AccessType.getId())) {
    auto CT = getTypeFromTBAAString(Id->getString().str(), I, MST);
    if (CT.isKnown()) {
      return TypeTree(CT).Only(-1, &I);
    }
  }

  llvm::SmallVector<TypeTree, 4> FieldTrees;
  for (unsigned i = 0, size = AccessType.getNumFields(); i < size; ++i) {
    auto SubAccess = AccessType.getFieldType(i);
    auto Offset = AccessType.getFieldOffset(i);
    auto SubResult = parseTBAA(SubAccess, I, DL, MST);
    FieldTrees.push_back(SubResult.ShiftIndices(DL, /*init offset*/ 0,
                                                /*max size*/ -1,
                                                /*addOffset*/ Offset));
  }

  return mergeNonConflictingTBAATrees(FieldTrees, DL);
}

/// Given a TBAA metadata node return the corresponding TypeTree
/// Modified from llvm::MDNode::isTBAAVtableAccess()
static inline TypeTree parseTBAA(const llvm::MDNode *M, llvm::Instruction &I,
                                 const llvm::DataLayout &DL,
                                 std::shared_ptr<llvm::ModuleSlotTracker> MST) {
  if (!isStructPathTBAA(M)) {
    if (M->getNumOperands() < 1)
      return TypeTree();
    if (const llvm::MDString *Tag1 =
            llvm::dyn_cast<llvm::MDString>(M->getOperand(0))) {
      return TypeTree(getTypeFromTBAAString(Tag1->getString().str(), I, MST))
          .Only(0, &I);
    }
    return TypeTree();
  }

  // For struct-path aware TBAA, we use the access type of the tag.
  TBAAStructTagNode Tag(M);
  if (!Tag.getAccessType())
    return TypeTree();
  TBAAStructTypeNode AccessType(Tag.getAccessType());
  return parseTBAA(AccessType, I, DL, MST);
}

/// Given an llvm::Instruction, return a TypeTree representing any
/// types that can be derived from TBAA metadata attached
static inline TypeTree parseTBAA(llvm::Instruction &I,
                                 const llvm::DataLayout &DL,
                                 std::shared_ptr<llvm::ModuleSlotTracker> MST) {
  llvm::SmallVector<TypeTree, 4> FieldTrees;
  if (const llvm::MDNode *M =
          I.getMetadata(llvm::LLVMContext::MD_tbaa_struct)) {
    for (unsigned i = 0, size = M->getNumOperands(); i < size; i += 3) {
      if (const llvm::MDNode *M2 =
              llvm::dyn_cast<llvm::MDNode>(M->getOperand(i + 2))) {
        auto SubResult = parseTBAA(M2, I, DL, MST);
        auto Start = llvm::cast<llvm::ConstantInt>(
                         llvm::cast<llvm::ConstantAsMetadata>(M->getOperand(i))
                             ->getValue())
                         ->getLimitedValue();
        auto Len =
            llvm::cast<llvm::ConstantInt>(
                llvm::cast<llvm::ConstantAsMetadata>(M->getOperand(i + 1))
                    ->getValue())
                ->getLimitedValue();
        FieldTrees.push_back(SubResult.ShiftIndices(DL, /*init offset*/ 0,
                                                    /*max size*/ Len,
                                                    /*add offset*/ Start));
      }
    }
  }
  if (const llvm::MDNode *M = I.getMetadata(llvm::LLVMContext::MD_tbaa)) {
    FieldTrees.push_back(parseTBAA(M, I, DL, MST));
  }
  return mergeNonConflictingTBAATrees(FieldTrees, DL);
}

#endif
