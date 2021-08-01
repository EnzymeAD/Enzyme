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

//static TypeTree getTypeTreeFromDITypeString(std::string TypeName, Instruction& I) {
//  if (TypeName[0] == '&') {
//    TypeTree TT = getTypeTreeFromDITypeString(TypeName.substr(1), I);
//    TypeTree Result(BaseType::Pointer);
//    Result |= TT;
//    return Result.Only(-1);
//  }
//  else {
//    ConcreteType CT = getConcreteTypeFromDITypeString(TypeName, I);
//    return TypeTree(CT).Only(-1);
//  }
//}

TypeTree parseDIType(DICompositeType& Type, Instruction& I, DataLayout& DL) {
  TypeTree Result;
  if (Type.getTag() & dwarf::DW_TAG_array_type) {
    DIType* SubType = Type.getBaseType();
    TypeTree SubTT = parseDIType(*SubType, I, DL);
    size_t Align = Type.getAlignInBytes();
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
          Result |= SubTT.ShiftIndices(DL, 0, -1, pos);
          pos += Align;
        }
      }
      else {
        assert(0 && "There shouldn't be non-constant-size arrays in Rust");
      }
    }
    return Result;
  }
  else {
    assert(0 && "Composite types other than array are not supported by Rust debug info parser");
  }
}

TypeTree parseDIType(DIDerivedType& Type, Instruction& I, DataLayout& DL) {
  if (Type.getTag() & dwarf::DW_TAG_pointer_type) {
    TypeTree Result(BaseType::Pointer);
    Result |= parseDIType(*Type.getBaseType(), I, DL);
    return Result.Only(-1);
  }
  else {
    assert(0 && "Derived types other than reference are not supported by Rust debug info parser");
  }
}

TypeTree parseDIType(DIType& Type, Instruction& I, DataLayout& DL) {
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
  return parseDIType(*type, I, DL);
}
