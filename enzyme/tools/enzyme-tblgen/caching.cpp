#include "caching.h"
// clang-format off
//

using namespace llvm;

// scalar (e.g xinc) is needed to be preserved if
// 1) it is potentially overwritten AND EITHER
//     a) x is active (for performing the shadow increment) or
//     b) we're not caching x and need xinc to compute the
//     derivative of a different variable
void emit_need_cache_info(const TGPattern &pattern, raw_ostream &os) {
  auto typeMap = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();
  const auto argUsers = pattern.getArgUsers();
  
  os 
<< "  // len, fp, etc. must be preserved if overwritten\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    // how about layout?
    if (ty == ArgType::cblas_layout) {
      continue;
    }
    auto name = nameVec[i];
    const auto users = argUsers.lookup(i);

    if (users.size() == 0) {
os << "  bool need_" << name << " = false;\n";
    } else {
      os 
<< "  bool need_" << name
<< " = ";
      bool first = true;
      for (size_t user: users) {
        auto userName = nameVec[user];
        if (name == userName) {
          llvm::errs() << pattern.getName() << "\n";
          llvm::errs() << "user: " << userName << "\n";
          PrintFatalError("scalar user is the same as the scalar itself");
          assert(false);
        }
        os << (first ? "" : " || ")
<< "active_" << userName;
        first = false;
      }
      os 
<< ";\n";
    }
  }
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    // how about layout?
    if (ty == ArgType::cblas_layout) {
      continue;
    }
    auto name = nameVec[i];
    os << "  bool cache_" << name << " = cacheMode";
    // scalars passed by value don't have to be cached
    if (!isVecLikeArg(ty))
      os << " && byRef";
    os << " && overwritten_" << name << " && need_" << name << ";\n";
  }
}

// TODO: maybe update to return set<StringRef>,
// for the case of multiple inputs
std::string get_input_mat(const DagInit *ruleDag) {
  std::string toCache = "";
  const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
  if (Def->isSubClassOf("Seq")) {
    // handle seq rules
    for (size_t i = 0; i < ruleDag->getNumArgs(); i++) {
      Init *subArg = ruleDag->getArg(i);
      DagInit *sub_Dag = cast<DagInit>(subArg);
      for (size_t j = 0; j < sub_Dag->getNumArgs(); j++) {
        Init *subArg = sub_Dag->getArg(j);
        if (DefInit *def = dyn_cast<DefInit>(subArg)) {
          const auto Def = def->getDef();
          if (Def->isSubClassOf("input")) {
            toCache = Def->getValueAsString("name");
            break;
          }
        }
      }
    }
  } else {
    for (size_t j = 0; j < ruleDag->getNumArgs(); j++) {
      Init *subArg = ruleDag->getArg(j);
      if (DefInit *def = dyn_cast<DefInit>(subArg)) {
        const auto Def = def->getDef();
        if (Def->isSubClassOf("input")) {
          toCache = Def->getValueAsString("name");
          break;
        }
      }
    }
  }
  return toCache;
}

void emit_input_caching(const TGPattern &pattern, raw_ostream &os) {
  // now we check for primal<X> usages, those must be cached,
  // if the corresponding rule is active
  auto rules = pattern.getRules();
  const auto nameVec = pattern.getArgNames();
  const auto activeArgs = pattern.getActiveArgs();
  assert(rules.size() == activeArgs.size());
  for (size_t i = 0; i < rules.size(); i++) {
    auto rule = rules[i];
    const auto activeArg = activeArgs[i];
    const auto name = nameVec[activeArg];
    const DagInit *ruleDag = rule.getRuleDag();
    // will update it directly in the next PR for nested rules
    std::string toCache = get_input_mat(ruleDag);
    if (toCache != "") {
      os << "  // we cache the following matrix,\n"
         << "  // since one rule uses input<" << toCache << ">\n"
         << "  if (active_" << name << ") {\n"
         << "    need_" << toCache << " = true;\n"
         << "    cache_" << toCache << " = true;\n"
         << "  }\n";
    }

  }
}

void emit_cacheTypes(const TGPattern &pattern, raw_ostream &os) {
  auto typeMap = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    const char* scalarType;
    if (ty == ArgType::len || ty == ArgType::vincInc || ty == ArgType::mldLD) {
      scalarType = "intType";
    } else if (ty == ArgType::trans || ty == ArgType::uplo ||
               ty == ArgType::diag || ty == ArgType::side) {      
      scalarType = "charType";
    } else if (ty == ArgType::fp) {
      scalarType = "fpType";
    } else {
      assert(ty == ArgType::cblas_layout || isVecLikeArg(ty));
      continue;
    }
  os
<< "  if (cache_" << nameVec[i] << ")\n"
<< "    cacheTypes.push_back(" << scalarType << ");\n";
  }
  // second loop, because we first cache scalars, then vectors
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (isVecLikeArg(ty)) {
      os
<< "  if (cache_" << nameVec[i] << ")\n"
<< "    cacheTypes.push_back(PointerType::getUnqual(fpType));\n";
    }
  }
}

void emit_vec_like_copy(const TGPattern &pattern, raw_ostream &os) {
  auto actArgs = pattern.getActiveArgs();
  auto typeMap = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();
  auto argUsers = pattern.getArgUsers();

  std::string valueTypes = "";
  { bool comma = false;
  for (auto i : nameVec) {
    if (comma) valueTypes += ", ";
    valueTypes += "ValueType::None";
    comma = true;
  }
  }

  for (size_t i = 0; i < actArgs.size(); i++) {
    size_t argIdx = actArgs[i];
    auto ty = typeMap.lookup(argIdx);
    if (ty != ArgType::ap && ty != ArgType::mldData && ty != ArgType::vincData)
      continue;
    auto name = nameVec[argIdx];
    auto dimensions = pattern.getRelatedLengthArgs(argIdx);

    if (ty == ArgType::ap) {
      os
<< "    if (cache_" << name << ") {\n"
<< "      Value *malloc_size;\n"
<< "      // arg_malloc_size will keep the original type\n"
<< "      Value *arg_malloc_size;\n"
<< "      malloc_size = arg_" << nameVec[dimensions[0]] << ";\n"
<< "      arg_malloc_size = malloc_size;\n"
<< "      malloc_size = load_if_ref(BuilderZ, intType, malloc_size, byRef);\n"
<< "      auto malins = CreateAllocation(BuilderZ, fpType, malloc_size, \"cache." << name << "\");\n";

    os
<< "      Value *margs[] = {malins, arg_" << name << ", malloc_size, llvm::ConstantInt::getFalse(IntegerType::getInt1Ty(call.getContext()))};\n"
<< "      Type *tys[] = {margs[0]->getType(), margs[1]->getType(),"
<< "                     margs[2]->getType()};\n"
<< "      auto memcpyF = Intrinsic::getDeclaration(gutils->oldFunc->getParent(), Intrinsic::memcpy, tys);\n"
<< "      BuilderZ.CreateCall(memcpyF, margs);\n"
<< "      cacheValues.push_back(malins);\n"
<< "    }\n";
  } else if (ty == ArgType::vincData) {
    assert(typeMap.lookup(argIdx+1) == ArgType::vincInc);
    auto vecName = nameVec[argIdx];
    auto incName = nameVec[argIdx+1];
    auto dimensions = pattern.getRelatedLengthArgs(argIdx);
    os
<< "    if (cache_" << vecName << ") {\n"
<< "      Value *malloc_size;\n"
<< "      // arg_malloc_size will keep the original type\n"
<< "      Value *arg_malloc_size;\n";

    if (dimensions.size() == 3) {
      os 
<< "      malloc_size = select_vec_dims(BuilderZ, arg_" << nameVec[dimensions[0]] << ", arg_" << nameVec[dimensions[1]] << ", arg_" << nameVec[dimensions[2]] << ", byRef);\n";
    } else {
      os 
<< "      malloc_size = arg_" << nameVec[dimensions[0]] << ";\n";
    }
    os
<< "      arg_malloc_size = malloc_size;\n"
<< "      malloc_size = load_if_ref(BuilderZ, intType, malloc_size, byRef);\n"
<< "      auto malins = CreateAllocation(BuilderZ, fpType, malloc_size, \"cache." << vecName << "\");\n"
<< "      ValueType valueTypes[] = {" << valueTypes << "};\n"
<< "      valueTypes[" << argIdx << "] = ValueType::Primal;\n"
<< "      if (byRef) valueTypes[" << argIdx+1 << "] = ValueType::Primal;\n";
    for (auto len_pos : pattern.getRelatedLengthArgs(argIdx) ) {
os << "      if (byRef) valueTypes[" << len_pos << "] = ValueType::Primal;\n";
    }
os << "      if (EnzymeBlasCopy) {\n"
<< "        Value *args[5] = {arg_malloc_size, arg_" << vecName << ", arg_" << incName << ", malins, to_blas_callconv(BuilderZ, ConstantInt::get(intType, 1), byRef, julia_decl_type, allocationBuilder)};\n"
<< "        callMemcpyStridedBlas(BuilderZ, *gutils->oldFunc->getParent(), blas, args, gutils->getInvertedBundles(&call, valueTypes, BuilderZ, /*lookup*/false));\n"
<< "      } else {\n"
<< "       auto dmemcpy = getOrInsertMemcpyStrided(*gutils->oldFunc->getParent(), fpType, cast<PointerType>(malins->getType()), intType, 0, 0);\n"
<< "        Value *inc = load_if_ref(BuilderZ, intType, arg_" << incName << ", byRef);\n"
<< "        Value *args[4] = {malins, arg_" << vecName << ", malloc_size, inc};\n"
<< "        if (args[1]->getType()->isIntegerTy())\n"
<< "          args[1] = BuilderZ.CreateIntToPtr(args[1], malins->getType());\n"
<< "        else if (args[1]->getType() != malins->getType())\n"
<< "          args[1] = BuilderZ.CreatePointerCast(args[1], malins->getType());\n"
<< "        BuilderZ.CreateCall(dmemcpy, args,\n"
<< "            gutils->getInvertedBundles(&call, valueTypes,\n"
<< "            BuilderZ, /*lookup*/ false));\n"
<< "      }\n"
<< "      cacheValues.push_back(malins);\n"
<< "    }\n";
  } else {
    assert(ty == ArgType::mldData);
    assert(typeMap.lookup(argIdx+1) == ArgType::mldLD);
    auto matName = nameVec[argIdx];
    auto ldName = nameVec[argIdx+1];
    auto dimensions = pattern.getRelatedLengthArgs(argIdx);
    std::string dim1, dim2;
    if (dimensions.size() == 2) {
      // mat is invariant to transpose
      dim1 = "arg_" + nameVec[dimensions[0]]; 
      dim2 = "arg_" + nameVec[dimensions[1]];
    } else {
      assert(dimensions.size() == 3);
      dim1 = "arg_" + nameVec[dimensions[1]];
      dim2 = "arg_" + nameVec[dimensions[2]];
    }
    os
<< "    if (cache_" << matName << ") {\n"
<< "      auto charTy = IntegerType::get(intType->getContext(), 8);\n"
<< "      Value *M, *N;\n";

    if (dimensions.size() == 3) {
      os 
<< "      Value *normal = is_normal(BuilderZ, arg_" << nameVec[dimensions[0]] << ", byRef);\n"
<< "      M = BuilderZ.CreateSelect(normal, " << dim1 << ", " << dim2 << ");\n"
<< "      N = BuilderZ.CreateSelect(normal, " << dim2 << ", " << dim1 << ");\n";
    } else {
      os 
<< "      M = " << dim1 << ";\n"
<< "      N = " << dim2 << ";\n";
    }

    os
<< "      auto *len1 = load_if_ref(BuilderZ, intType, M, byRef);\n"
<< "      auto *len2 = load_if_ref(BuilderZ, intType, N, byRef);\n"
<< "      auto *matSize = BuilderZ.CreateMul(len1, len2);\n"
<< "      auto malins = CreateAllocation(BuilderZ, fpType, matSize, \"cache." << matName << "\");\n"
<< "      ValueType valueTypes[] = {" << valueTypes << "};\n"
<<"       valueTypes[" << argIdx << "] = ValueType::Primal;\n"
<< "      if (byRef) valueTypes[" << argIdx+1 << "] = ValueType::Primal;\n";
    for (auto len_pos : dimensions ) {
os << "      if (byRef) valueTypes[" << len_pos << "] = ValueType::Primal;\n";
    }
os << "      if (EnzymeLapackCopy) {\n"
<< "        Value *uplo = llvm::ConstantInt::get(charTy, 0);\n" // garbage data, just should not match U or L
<< "        uplo = to_blas_callconv(BuilderZ, uplo, byRef, nullptr, allocationBuilder, \"copy.garbage\");\n"
<< "        Value *args[7] = {uplo, M, N, arg_" << matName << ", arg_" << ldName << ", malins, M};\n"
<< "        callMemcpyStridedLapack(BuilderZ, *gutils->oldFunc->getParent(), blas, args, gutils->getInvertedBundles(&call, valueTypes, BuilderZ, /*lookup*/false));\n"
<< "      } else {\n"
<< "        auto dmemcpy = getOrInsertMemcpyMat(*gutils->oldFunc->getParent(), fpType, cast<PointerType>(malins->getType()), intType, 0, 0);\n"
<< "        Value *len_lda = load_if_ref(BuilderZ, intType, arg_" << ldName << ", byRef);\n"
<< "        Value *args[5] = {malins, arg_" << matName << ", len1, len2, len_lda};\n"
<< "        if (args[1]->getType()->isIntegerTy())\n"
<< "          args[1] = BuilderZ.CreateIntToPtr(args[1], malins->getType());\n"
<< "        else if (args[1]->getType() != malins->getType())\n"
<< "          args[1] = BuilderZ.CreatePointerCast(args[1], malins->getType());\n"
<< "        BuilderZ.CreateCall(dmemcpy, args,\n"
<< "            gutils->getInvertedBundles(&call, valueTypes,\n"
<< "            BuilderZ, /*lookup*/ false));\n"
<< "      }\n"
<< "      cacheValues.push_back(malins);\n"
<< "    }\n";
    }
  }
}

void emit_cache_for_reverse(const TGPattern &pattern, raw_ostream &os) {
  auto typeMap = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();
  auto argUsers = pattern.getArgUsers();
  const auto activeArgs = pattern.getActiveArgs();
  auto rules = pattern.getRules();

  os 
<< "  if ((Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "       Mode == DerivativeMode::ReverseModePrimal) && cachetype) {\n"
<< "    SmallVector<Value *, 2> cacheValues;\n";
 
  os << "    if (byRef) {\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    auto name = nameVec[i];

    const char* scalType;
    if (ty == ArgType::len || ty == ArgType::vincInc || ty == ArgType::mldLD) {
      scalType = "intType";
    } else if (ty == ArgType::trans) {
      scalType = "charType";
    } else if (ty == ArgType::fp) {
      scalType = "fpType";
    } else {
      continue;
    }
    os 
<< "        addValueToCache(arg_" << name <<", cache_" << name <<", "  
<< scalType << ", cacheValues, BuilderZ, \"" << name << "\");\n";
  }
  os << "    }\n";

  // handle vec, ap and mat
  emit_vec_like_copy(pattern, os);

  os
<< "    if (cacheValues.size() == 1) {\n"
<< "      cacheval = cacheValues[0];\n"
<< "    } else {\n"
<< "      cacheval = UndefValue::get(cachetype);\n"
<< "      for (auto&& tup : llvm::enumerate(cacheValues))\n"
<< "        cacheval = BuilderZ.CreateInsertValue(cacheval, tup.value(), tup.index());\n"
<< "    }\n"
<< "    gutils->cacheForReverse(BuilderZ, cacheval,\n"
<< "                            getIndex(&call, CacheType::Tape));\n"
<< "  }\n"
<< "  unsigned cacheidx = 0;\n";

  assert(rules.size() == activeArgs.size());
  for (size_t a = 0; a < rules.size(); a++) {
    auto rule = rules[a];
    auto i = activeArgs[a];
    auto name = nameVec[i];
    auto ty = typeMap.lookup(i);
    if (ty == ArgType::vincData) {
      assert(typeMap.lookup(i+1) == ArgType::vincInc);
      auto vecName = nameVec[i];
      auto incName = nameVec[i+1];
      os
<< "  Value *true_" << incName << " = arg_" << incName << ";\n"
<< "  Value *free_" << vecName << " = nullptr;\n";
    } else if (ty == ArgType::ap) {
      auto apName = nameVec[i];
      os << "  Value *free_" << apName << " = nullptr;\n";      
    } else if (ty == ArgType::mldData) {
      assert(typeMap.lookup(i+1) == ArgType::mldLD);
      auto vecName = nameVec[i];
      auto ldName = nameVec[i+1];
      os
<< "  Value *true_" << ldName << " = arg_" << ldName << ";\n"
<< "  Value *" << ldName << " = true_" << ldName << ";\n"
<< "  Value *free_" << vecName << " = nullptr;\n";
     
    }

    const DagInit *ruleDag = rule.getRuleDag();
    std::string toCache = get_input_mat(ruleDag);
    if (toCache != "") {
      os << "  Value *input_" << toCache << " = nullptr;\n"
         << "  Value *free_input_" << toCache << " = nullptr;\n";
    }
  }

  os
<< "  IRBuilder<> Builder2(&call);\n"               
<< "  switch (Mode) {\n"                            
<< "    case DerivativeMode::ReverseModeCombined:\n"
<< "    case DerivativeMode::ReverseModeGradient:\n"
<< "      getReverseBuilder(Builder2);\n"
<< "      break;\n"
<< "    case DerivativeMode::ForwardMode:\n"
<< "    case DerivativeMode::ForwardModeSplit:\n"
<< "      Builder2.SetInsertPoint(BuilderZ.GetInsertBlock(),\n"
<< "                              BuilderZ.GetInsertPoint());\n"
<< "      Builder2.setFastMathFlags(getFast());\n"
<< "      break;\n"
<< "    case DerivativeMode::ReverseModePrimal:\n"
<< "      break;\n"
<< "  }\n\n";
}

void emit_caching(const TGPattern &pattern, raw_ostream &os) {

  auto actArgs = pattern.getActiveArgs();
  auto nameVec = pattern.getArgNames();
  const auto typeMap = pattern.getArgTypeMap();

  // 1. No caching for fwd-mode
  // 2. Deactivate caching for uncacheable_args
  // 3. Only caching if we do need the primary for an active gradient.
  // 4. (New) Cache vec if it is overwritten but the input vec is required.
  os 
<< "  SmallVector<Type *, 2> cacheTypes;\n\n";

  emit_need_cache_info(pattern, os);
  emit_input_caching(pattern, os);
  emit_cacheTypes(pattern, os);

  os
<< "  Type *cachetype = nullptr;\n"
<< "  switch (cacheTypes.size()) {\n"
<< "  case 0:\n"
<< "    break;\n"
<< "  case 1:\n"
<< "    cachetype = cacheTypes[0];\n"
<< "    break;\n"
<< "  default:\n"
<< "    cachetype = StructType::get(call.getContext(), cacheTypes);\n"
<< "    break;\n"
<< "  }\n\n";

  emit_cache_for_reverse(pattern, os);
}
