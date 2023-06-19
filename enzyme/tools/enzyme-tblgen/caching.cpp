#include "caching.h"
// clang-format off
//


using namespace llvm;
void emit_mat_vec_caching(TGPattern &pattern, size_t i, raw_ostream &os) {
  const auto argUsers = pattern.getArgUsers();
  const auto nameVec = pattern.getArgNames();
  const auto name = nameVec[i];
  const auto vecPosition = i;
  const auto vecUsers = argUsers.lookup(vecPosition);
    if (vecUsers.size() == 0) {
os << "  bool need_" << name << " = false;\n";
    } else {
      os 
<< "  bool need_" << name
<< "    = ";
      bool first = true;
      for (size_t user: vecUsers) {
        auto userName = nameVec[user];
        if (name == userName)
          continue; // adjoint of x won't need x
        os << (first ? "" : " || ")
<< "active_" << userName;
        first = false;
      }
      os 
<< ";\n";
    }
    os 
<< "  bool cache_" << name << " = cacheMode && overwritten_" << name << " && need_" << name << ";\n";
}

// scalar (e.g xinc) is needed to be preserved if
// 1) it is potentially overwritten AND EITHER
//     a) x is active (for performing the shadow increment) or
//     b) we're not caching x and need xinc to compute the
//     derivative of a different variable
void emit_scalar_caching(TGPattern &pattern, raw_ostream &os) {
  auto actArgs = pattern.getActiveArgs();
  auto typeMap = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();
  const auto argUsers = pattern.getArgUsers();
  
  os 
<< "  // len, fp, etc. must be preserved if overwritten\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != argType::len && ty != argType::fp && ty != argType::trans 
        && ty != argType::vincInc && ty != argType::mldLD) {
      continue;
    }
    auto name = nameVec[i];
    const auto scalarUsers = argUsers.lookup(i);

    if (scalarUsers.size() == 0) {
os << "  bool need_" << name << " = false;\n";
    } else {
      os 
<< "  bool need_" << name
<< " = ";
      bool first = true;
      for (size_t user: scalarUsers) {
        auto userName = nameVec[user];
        //if (name == userName)
        //  continue; // adjoint of x won't need x
        os << (first ? "" : " || ")
<< "active_" << userName;
        first = false;
      }
      os 
<< ";\n";
    }
    os 
<< "  bool cache_" << name << " = cacheMode && byRef && overwritten_" << name << " && need_" << name << ";\n";

  }
}
void emit_scalar_cacheTypes(TGPattern &pattern, raw_ostream &os) {
  auto actArgs = pattern.getActiveArgs();
  auto typeMap = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    auto scalarType = "";
    if (ty == argType::len || ty == argType::vincInc || ty == argType::mldLD) {
      scalarType = "intType";
    } else if (ty == argType::trans) {
      scalarType = "charType";
    } else if (ty == argType::fp) {
      scalarType = "fpType";
    } else {
      continue;
    }
  os
<< "  if (cache_" << nameVec[i] << ")\n"
<< "    cacheTypes.push_back(" << scalarType << ");\n";
  }
}

void emit_vec_copy(TGPattern &pattern, raw_ostream &os) {
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
    if (ty != argType::vincData)
      continue;
    assert(typeMap.lookup(argIdx+1) == argType::vincInc);
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
<< "      if (byRef) {\n"
<< "        malloc_size = BuilderZ.CreateLoad(intType, BuilderZ.CreatePointerCast(malloc_size, PointerType::get(intType, cast<PointerType>(malloc_size->getType())->getAddressSpace())));\n"
<< "      }\n"
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
<< "        Value *inc = arg_" << incName << ";\n"
<< "        if (byRef) {\n"
<< "          auto tmp = BuilderZ.CreatePointerCast(inc, PointerType::get(intType, cast<PointerType>(inc->getType())->getAddressSpace()));\n"
<< "          inc = BuilderZ.CreateLoad(intType, tmp);\n"
<< "        }\n"
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
  }
}

void emit_mat_copy(TGPattern &pattern, raw_ostream &os) {
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
    if (ty != argType::mldData)
      continue;
    assert(typeMap.lookup(argIdx+1) == argType::mldLD);
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
<< "      Value *matSize;\n"
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
<< "      auto *len1 = M;\n"
<< "      auto *len2 = N;\n"
<< "      if (byRef) {\n"
<< "        auto MP = BuilderZ.CreatePointerCast(M, PointerType::get(intType, cast<PointerType>(M->getType())->getAddressSpace()));\n"
<< "        auto NP = BuilderZ.CreatePointerCast(N, PointerType::get(intType, cast<PointerType>(N->getType())->getAddressSpace()));\n"
<< "        len1 = BuilderZ.CreateLoad(intType, MP);\n"
<< "        len2 = BuilderZ.CreateLoad(intType, NP);\n"
<< "        matSize = BuilderZ.CreateMul(len1, len2);\n"
<< "      } else {\n"
<< "        matSize = BuilderZ.CreateMul(M,N);\n"
<< "      }\n"
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
<< "        Value *len_lda = arg_" << ldName << ";\n"
<< "        if (byRef) {\n"
<< "          auto LDP = BuilderZ.CreatePointerCast(len_lda, PointerType::get(intType, cast<PointerType>(len_lda->getType())->getAddressSpace()));\n"
<< "          len_lda = BuilderZ.CreateLoad(intType, LDP);\n"
<< "        }\n"
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


void emit_cache_for_reverse(TGPattern &pattern, raw_ostream &os) {
  auto actArgs = pattern.getActiveArgs();
  auto typeMap = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();
  auto argUsers = pattern.getArgUsers();

  os 
<< "  if ((Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "       Mode == DerivativeMode::ReverseModePrimal) && cachetype) {\n"
<< "    SmallVector<Value *, 2> cacheValues;\n";
 
  os << "    if (byRef) {\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    auto name = nameVec[i];

    std::string scalType = "";
    if (ty == argType::len || ty == argType::vincInc || ty == argType::mldLD) {
      scalType = "intType";
    } else if (ty == argType::trans) {
      scalType = "charType";
    } else if (ty == argType::fp) {
      scalType = "fpType";
    } else {
      continue;
    }
    os 
<< "        addValueToCache(arg_" << name <<", cache_" << name <<", "  
<< scalType << ", cacheValues, BuilderZ, \"" << name << "\");\n";
  }
  os << "    }\n";

  emit_mat_copy(pattern, os);
  emit_vec_copy(pattern, os);

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

  // following code is just leftovers
  // once cleaned up, at most free_ args should be left
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto name = nameVec[i];
    auto ty = typeMap.lookup(i);
    if (ty == argType::vincData) {
      assert(typeMap.lookup(i+1) == argType::vincInc);
      auto vecName = nameVec[i];
      auto incName = nameVec[i+1];
      os 
<< "  Value *true_" << incName << " = arg_" << incName << ";\n"
<< "  Value *free_" << vecName << " = nullptr;\n";
    } else if (ty == argType::mldData) {
      assert(typeMap.lookup(i+1) == argType::mldLD);
      auto vecName = nameVec[i];
      auto ldName = nameVec[i+1];
      os 
<< "  Value *true_" << ldName << " = arg_" << ldName << ";\n"
<< "  Value *" << ldName << " = true_" << ldName << ";\n"
<< "  Value *free_" << vecName << " = nullptr;\n";
    } else if (ty == argType::len) {
    } else if (ty == argType::fp) {
    } else if (ty == argType::trans) {
    }
  }


  os
<< "  IRBuilder<> Builder2(call.getParent());\n"
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

void emit_caching(TGPattern &pattern, raw_ostream &os) {

  auto actArgs = pattern.getActiveArgs();
  auto nameVec = pattern.getArgNames();
  const auto typeMap = pattern.getArgTypeMap();

  // 1. No caching for fwd-mode
  // 2. Deactivate caching for uncacheable_args
  // 3. Only caching if we do need the primary for an active gradient.
  // 4. (New) Cache vec if it is overwritten but the input vec is required.
  os 
<< "  SmallVector<Type *, 2> cacheTypes;\n\n";

  emit_scalar_caching(pattern, os);
  emit_scalar_cacheTypes(pattern, os);
  // we currently cache all vecs before we cache all matrices
  // once fixed we can merge this calls
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != argType::vincData)
      continue;
    assert(typeMap.lookup(i+1) == argType::vincInc);
    emit_mat_vec_caching(pattern, i, os);
  }
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != argType::mldData)
      continue;
    assert(typeMap.lookup(i+1) == argType::mldLD);
    emit_mat_vec_caching(pattern, i, os);
  }

  for (auto&& actEn : llvm::enumerate(actArgs)) {
    if (typeMap.lookup(actEn.value()) == argType::fp) continue;
    auto name = nameVec[actEn.value()];
    os 
<< "  if (cache_" << name << ")\n"
<< "    cacheTypes.push_back(PointerType::getUnqual(fpType));\n";
  }
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


