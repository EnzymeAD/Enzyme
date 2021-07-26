//
// Created by Chuyang Chen on 14/7/2021.
//

#ifndef ENZYME_RUSTDEBUGINFO_H
#define ENZYME_RUSTDEBUGINFO_H 1

static inline ConcreteType getConcreteTypeFromDITypeString(std::string TypeName, Instruction& I) {
  if (TypeName == "f64") {
    return Type::getDoubleTy(I.getContext());
  }
  else if (TypeName == "f32") {
    return Type::getFloatTy(I.getContext());
  }
  else if (TypeName == "i8" || TypeName == "i16" ||TypeName == "i32" || TypeName == "i64" || TypeName == "isize" ||
           TypeName == "u8" || TypeName == "u16" ||TypeName == "u32" || TypeName == "u64" || TypeName == "usize") {
    return ConcreteType(BaseType::Integer);
  }
  else if (TypeName[0] == '&'){
    return ConcreteType(BaseType::Pointer);
  }
  else {
    return ConcreteType(BaseType::Unknown);
  }
}

static inline TypeTree parseDIType(DbgDeclareInst& I) {
  DIType* type = I.getVariable()->getType();
  auto CT = getConcreteTypeFromDITypeString(type->getName().str(), I);
  if (!CT.isPossiblePointer()) {
    return TypeTree(CT).Only(-1);
  }
  else {
//    TypeTree Result(BaseType::Pointer);
//    Result.Only(-1);
//    Result |= TypeTree(Type::getDoubleTy(I.getContext())).Only(-1);
//    return Result.Only(-1);
    return TypeTree(BaseType::Unknown).Only(-1);
  }
}

#endif //ENZYME_RUSTDEBUGINFO_H
