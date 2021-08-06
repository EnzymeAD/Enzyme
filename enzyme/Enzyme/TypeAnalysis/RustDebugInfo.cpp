//
// Created by Chuyang Chen on 30/7/2021.
//
#include <tuple>

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/CommandLine.h"

#include "RustDebugInfo.h"

using std::tuple;
using std::get;
using std::make_tuple;

TypeTree parseDIType(DIType& Type, Instruction& I, DataLayout& DL);

TypeTree parseDIType(DIBasicType& Type, Instruction& I, DataLayout& DL) {
  std::string TypeName = Type.getName().str();
  TypeTree Result;
  size_t Size = Type.getSizeInBits() / 8;
  if (TypeName == "f64") {
    Result = TypeTree(Type::getDoubleTy(I.getContext())).Only(0);
  }
  else if (TypeName == "f32") {
    Result = TypeTree(Type::getFloatTy(I.getContext())).Only(0);
  }
  else if (TypeName == "i8" || TypeName == "i16" ||TypeName == "i32" || TypeName == "i64" || TypeName == "isize" ||
           TypeName == "u8" || TypeName == "u16" ||TypeName == "u32" || TypeName == "u64" || TypeName == "usize") {
    Result = TypeTree(ConcreteType(BaseType::Integer)).Only(0);
  }
  else {
    Result = TypeTree(ConcreteType(BaseType::Unknown)).Only(0);
  }
  return Result;
}

TypeTree parseDIType(DICompositeType& Type, Instruction& I, DataLayout& DL) {
  TypeTree Result;
  if (Type.getTag() == dwarf::DW_TAG_array_type) {
    DIType* SubType = Type.getBaseType();
    TypeTree SubTT = parseDIType(*SubType, I, DL);
    size_t Align = Type.getAlignInBytes();
    size_t SubSize = SubType->getSizeInBits() / 8;
    size_t Size = Type.getSizeInBits() / 8;
    DINodeArray Subranges = Type.getElements();
    size_t pos = 0;
    for (auto r: Subranges) {
      DISubrange* Subrange = dyn_cast<DISubrange>(r);
      if (auto Count = Subrange->getCount().get<ConstantInt*>()) {
        int64_t count = Count->getSExtValue();
        if (count == -1) {
          break;
        }
        for (int64_t i = 0; i < count; i++) {
          Result |= SubTT.ShiftIndices(DL, 0, Size, pos);
          size_t tmp = pos + SubSize;
          if (tmp % Align != 0) {
            pos = (tmp / Align + 1) * Align;
          }
          else {
            pos = tmp;
          }
        }
      }
      else {
        assert(0 && "There shouldn't be non-constant-size arrays in Rust");
      }
    }
    return Result;
  }
  else if (Type.getTag() == dwarf::DW_TAG_structure_type) {
    DINodeArray Elements = Type.getElements();
    size_t Size = Type.getSizeInBits() / 8;
    for (auto e: Elements) {
      DIType *SubType = dyn_cast<DIDerivedType>(e);
      assert(SubType->getTag() == dwarf::DW_TAG_member);
      TypeTree SubTT = parseDIType(*SubType, I, DL);
      size_t Offset = SubType->getOffsetInBits() / 8;
      SubTT = SubTT.ShiftIndices(DL, 0, Size, Offset);
      Result |= SubTT;
    }
    return Result;
  }
  else {
    assert(0 && "Composite types other than array and struct are not supported by Rust debug info parser");
  }
}

TypeTree parseDIType(DIDerivedType& Type, Instruction& I, DataLayout& DL) {
  if (Type.getTag() == dwarf::DW_TAG_pointer_type) {
    TypeTree Result(BaseType::Pointer);
    DIType* SubType = Type.getBaseType();
    TypeTree SubTT = parseDIType(*SubType, I, DL);
    Result |= SubTT;
    return Result.Only(0);
  }
  else if (Type.getTag() == dwarf::DW_TAG_member) {
    DIType* SubType = Type.getBaseType();
    TypeTree Result = parseDIType(*SubType, I, DL);
    return Result;
  }
  else {
    assert(0 && "Derived types other than pointer and member are not supported by Rust debug info parser");
  }
}

TypeTree parseDIType(DIType& Type, Instruction& I, DataLayout& DL) {
  if (Type.getSizeInBits() == 0) {
    return TypeTree();
  }

  if (auto BT = dyn_cast<DIBasicType>(&Type)) {
    return parseDIType(*BT, I, DL);
  }
  else if (auto CT = dyn_cast<DICompositeType>(&Type)) {
    return parseDIType(*CT, I, DL);
  }
  else if (auto DT = dyn_cast<DIDerivedType>(&Type)) {
    return parseDIType(*DT, I, DL);
  }
  else {
    assert(0 && "Types other than floating-points, integers, arrays, and pointers are not supported by debug info parser");
  }
}

TypeTree parseDIType(DbgDeclareInst& I, DataLayout& DL) {
  DIType* type = I.getVariable()->getType();
  TypeTree Result = parseDIType(*type, I, DL);
//  TypeTree Debug = TypeTree(BaseType::Pointer);
//  Debug |= TypeTree(BaseType::Integer).Only(0);
//  Debug = Debug.Only(0);
  return Result;
}
