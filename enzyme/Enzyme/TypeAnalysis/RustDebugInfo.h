//
// Created by Chuyang Chen on 14/7/2021.
//

#ifndef ENZYME_RUSTDEBUGINFO_H
#define ENZYME_RUSTDEBUGINFO_H 1

static inline Optional<ConcreteType> getConcreteTypeFromDITypeString(std::string TypeName, Instruction& I) {
  if (TypeName == "f64") {
    return Optional<ConcreteType>(Type::getDoubleTy(I.getContext()));
  }
  else if (TypeName == "f32") {
    return Optional<ConcreteType>(Type::getFloatTy(I.getContext()));
  }
  else if (TypeName == "i8" || TypeName == "i16" ||TypeName == "i32" || TypeName == "i64" || TypeName == "isize" ||
           TypeName == "u8" || TypeName == "u16" ||TypeName == "u32" || TypeName == "u64" || TypeName == "usize") {
    return Optional<ConcreteType>(ConcreteType(BaseType::Integer));
  }
  else {
    // TODO[Chen] Consider pointers
    return Optional<ConcreteType>(ConcreteType(BaseType::Unknown));
  }
}

static inline TypeTree parseDIType(DbgDeclareInst& I) {
  DIType* type = I.getVariable()->getType();
  Optional<ConcreteType> CT = getConcreteTypeFromDITypeString(type->getName().str(), I);
  if (CT) {
    return TypeTree(CT.getValue()).Only(0);
  }
  else {
    // TODO[Chen] Consider non-concrete types
    assert(false);
  }
}

#endif //ENZYME_RUSTDEBUGINFO_H
