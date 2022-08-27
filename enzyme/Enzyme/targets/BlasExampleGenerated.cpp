/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

struct BlasInfo {
  StringRef floatType;
  StringRef prefix;
  StringRef suffix;
  StringRef function;
};

llvm::Optional<BlasInfo> extractBLAS(StringRef in) {
  llvm::Twine floatType[] = {"s", "d"}; // c, z
  llvm::Twine extractable[] = {"axpy", "dot", "scal"};
  llvm::Twine prefixes[] = {"", "cblas_", "cublas_"};
  llvm::Twine suffixes[] = {"", "_", "_64_"};
  for (auto t : floatType) {
    for (auto f : extractable) {
      for (auto p : prefixes) {
        for (auto s : suffixes) {
          if (in == (p + t + f + s).str()) {
            return llvm::Optional<BlasInfo>(BlasInfo{
                t.getSingleStringRef(),
                p.getSingleStringRef(),
                s.getSingleStringRef(),
                f.getSingleStringRef(),
            });
          }
        }
      }
    }
  }
  return llvm::NoneType();
}

bool handleBLAS(llvm::CallInst &call, Function *called, BlasInfo blas,
                const std::map<Argument *, bool> &uncacheable_args) { 
                                                                      
  bool result = true;                                                 
  if (!gutils->isConstantInstruction(&call)) {                        
    Type *fpType;                                                  
    if (blas.floatType == "d") {                                    
      fpType = Type::getDoubleTy(call.getContext());               
    } else if (blas.floatType == "s") {                             
      fpType = Type::getFloatTy(call.getContext());                
    } else {                                                          
      assert(false && "Unreachable");                               
    }                                                                 
     if (blas.function == "axpy") {                           
      result = handle_axpy(blas, call, called, uncacheable_args, fpType);                    
    } else  if (blas.function == "dot") {                           
      result = handle_dot(blas, call, called, uncacheable_args, fpType);                    
    } else  if (blas.function == "scal") {                           
      result = handle_scal(blas, call, called, uncacheable_args, fpType);                    
    } else {                                                          
      llvm::errs() << " fallback?\n";                              
      return false;                                                   
    }                                                                 
  }                                                                   
                                                                      
  if (Mode == DerivativeMode::ReverseModeGradient) {                  
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);             
  } else {                                                            
    eraseIfUnused(call);                                              
  }                                                                   
                                                                      
  return result;                                                      
}                                                                     

bool handle_axpy(BlasInfo blas, llvm::CallInst &call, Function *called,
    const std::map<Argument *, bool> &uncacheable_args, Type *fpType) {
  
  CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
  IRBuilder<> BuilderZ(newCall);
  BuilderZ.setFastMathFlags(getFast());
  IRBuilder<> allocationBuilder(gutils->inversionAllocs);
  allocationBuilder.setFastMathFlags(getFast());
  auto &DL = gutils->oldFunc->getParent()->getDataLayout();
  auto arg_n = call.getArgOperand(0);
  auto type_n = arg_n->getType();
  bool uncacheable_n = uncacheable_args.find(called->getArg(0))->second;

  auto arg_alpha = call.getArgOperand(1);
  auto type_alpha = arg_alpha->getType();
  bool uncacheable_alpha = uncacheable_args.find(called->getArg(1))->second;
  bool active_alpha = !gutils->isConstantValue(arg_alpha);

  auto arg_x = call.getArgOperand(2);
  auto type_x = arg_x->getType();
  bool uncacheable_x = uncacheable_args.find(called->getArg(2))->second;
  bool active_x = !gutils->isConstantValue(arg_x);

  auto arg_incx = call.getArgOperand(3);
  auto type_incx = arg_incx->getType();
  bool uncacheable_incx = uncacheable_args.find(called->getArg(3))->second;

  auto arg_y = call.getArgOperand(4);
  auto type_y = arg_y->getType();
  bool uncacheable_y = uncacheable_args.find(called->getArg(4))->second;
  bool active_y = !gutils->isConstantValue(arg_y);

  auto arg_incy = call.getArgOperand(5);
  auto type_incy = arg_incy->getType();
  bool uncacheable_incy = uncacheable_args.find(called->getArg(5))->second;

  /* beginning castvalls */
  Type *castvals[3];
  if (auto PT = dyn_cast<PointerType>(type_alpha))
    castvals[0] = PT;
  else
    castvals[0] = PointerType::getUnqual(fpType);
  if (auto PT = dyn_cast<PointerType>(type_x))
    castvals[1] = PT;
  else
    castvals[1] = PointerType::getUnqual(fpType);
  if (auto PT = dyn_cast<PointerType>(type_y))
    castvals[2] = PT;
  else
    castvals[2] = PointerType::getUnqual(fpType);
  Value *cacheval;

  /* ending castvalls */
  IntegerType *intType = dyn_cast<IntegerType>(type_n);
  bool byRef = false;
  if (!intType) {
    auto PT = cast<PointerType>(type_n);
    if (blas.suffix.contains("64"))
      intType = IntegerType::get(PT->getContext(), 64);
    else
      intType = IntegerType::get(PT->getContext(), 32);
    byRef = true;
  }

  SmallVector<Type *, 2> cacheTypes;

  // len, fp must be preserved if overwritten
  bool cache_n = false;
  if (byRef && uncacheable_n) {
    cacheTypes.push_back(intType);
    cache_n = true;
  }
  bool cache_alpha = false;
  if (byRef && uncacheable_alpha) {
    cacheTypes.push_back(fpType);
    cache_alpha = true;
  }
  bool cache_x  = Mode != DerivativeMode::ForwardMode &&
          uncacheable_x && active_alpha;
  bool cache_incx = false;
  bool need_incx = (active_x  || (!cache_x && (active_alpha)));
  if (byRef && uncacheable_incx && need_incx) {
    cacheTypes.push_back(intType);
    cache_incx = true;
   }

  bool cache_y  = Mode != DerivativeMode::ForwardMode &&
          uncacheable_y;
  bool cache_incy = false;
  bool need_incy = (active_y  || (!cache_y && (active_alpha || active_x)));
  if (byRef && uncacheable_incy && need_incy) {
    cacheTypes.push_back(intType);
    cache_incy = true;
   }

  int numCached = (int) cache_x + (int) cache_y;
  if (cache_alpha)
    cacheTypes.push_back(castvals[0]);
  if (cache_x)
    cacheTypes.push_back(castvals[1]);
  if (cache_y)
    cacheTypes.push_back(castvals[2]);
  Type *cachetype = nullptr;
  switch (cacheTypes.size()) {
  case 0:
    break;
  case 1:
    cachetype = cacheTypes[0];
    break;
  default:
    cachetype = StructType::get(call.getContext(), cacheTypes);
    break;
  }

  if ((Mode == DerivativeMode::ReverseModeCombined ||
       Mode == DerivativeMode::ReverseModePrimal) && cachetype) {
    SmallVector<Value *, 2> cacheValues;
    Value *len_n = gutils->getNewFromOriginal(arg_n);
    if (byRef) {
      len_n = BuilderZ.CreatePointerCast(len_n, PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
      len_n = BuilderZ.CreateLoad(intType, len_n);
#else
      len_n = BuilderZ.CreateLoad(len_n);
#endif
      if (cache_n)
        cacheValues.push_back(len_n);
    }
    Value *fp_alpha = gutils->getNewFromOriginal(arg_alpha);
    if (byRef) {
      fp_alpha = BuilderZ.CreatePointerCast(fp_alpha, PointerType::getUnqual(fpType));
#if LLVM_VERSION_MAJOR > 7
      fp_alpha = BuilderZ.CreateLoad(fpType, fp_alpha);
#else
      fp_alpha = BuilderZ.CreateLoad(fp_alpha);
#endif
      if (cache_alpha)
        cacheValues.push_back(fp_alpha);
    }
    Value *incx = gutils->getNewFromOriginal(arg_incx);
    if (byRef) {
      incx = BuilderZ.CreatePointerCast(incx, PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
      incx = BuilderZ.CreateLoad(intType, incx);
#else
      incx = BuilderZ.CreateLoad(incx);
#endif
      if (cache_incx)
        cacheValues.push_back(incx);
    }
    Value *incy = gutils->getNewFromOriginal(arg_incy);
    if (byRef) {
      incy = BuilderZ.CreatePointerCast(incy, PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
      incy = BuilderZ.CreateLoad(intType, incy);
#else
      incy = BuilderZ.CreateLoad(incy);
#endif
      if (cache_incy)
        cacheValues.push_back(incy);
    }
    if (cache_x) {
      auto dmemcpy = getOrInsertMemcpyStrided(
          *gutils->oldFunc->getParent(), cast<PointerType>(castvals[0]),
          type_n, 0, 0);
      auto malins = CreateAllocation(BuilderZ, fpType, len_n);
      Value *arg = BuilderZ.CreateBitCast(malins, castvals[0]);
      Value *args[4] = {arg,
                         gutils->getNewFromOriginal(arg_x),
                         len_n, incx};
      if (args[1]->getType()->isIntegerTy())
        args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[0]);
      BuilderZ.CreateCall(dmemcpy, args,
          gutils->getInvertedBundles(&call,{
ValueType::None, ValueType::None, ValueType::Shadow, ValueType::None, ValueType::None, ValueType::None},
          BuilderZ, /*lookup*/ false));
      cacheValues.push_back(arg);
    }
    if (cache_y) {
      auto dmemcpy = getOrInsertMemcpyStrided(
          *gutils->oldFunc->getParent(), cast<PointerType>(castvals[1]),
          type_n, 0, 0);
      auto malins = CreateAllocation(BuilderZ, fpType, len_n);
      Value *arg = BuilderZ.CreateBitCast(malins, castvals[1]);
      Value *args[4] = {arg,
                         gutils->getNewFromOriginal(arg_y),
                         len_n, incy};
      if (args[1]->getType()->isIntegerTy())
        args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[1]);
      BuilderZ.CreateCall(dmemcpy, args,
          gutils->getInvertedBundles(&call,{
ValueType::None, ValueType::None, ValueType::None, ValueType::None, ValueType::Shadow, ValueType::None},
          BuilderZ, /*lookup*/ false));
      cacheValues.push_back(arg);
    }
    if (cacheValues.size() == 1) {
      cacheval = cacheValues[0];
    } else {
      cacheval = UndefValue::get(cachetype);
      for (auto tup : llvm::enumerate(cacheValues))
        cacheval = BuilderZ.CreateInsertValue(cacheval, tup.value(), tup.index());
    }
    gutils->cacheForReverse(BuilderZ, cacheval,
                            getIndex(&call, CacheType::Tape));
  }
  unsigned cacheidx = 0;
  Value *len_n = gutils->getNewFromOriginal(arg_n);
  Value *fp_alpha = gutils->getNewFromOriginal(arg_alpha);
  Value *true_incx = gutils->getNewFromOriginal(arg_incx);
  Value *incx = true_incx;
  Value *data_x = gutils->getNewFromOriginal(arg_x);
  Value *data_ptr_x = nullptr;
  Value *true_incy = gutils->getNewFromOriginal(arg_incy);
  Value *incy = true_incy;
  Value *data_y = gutils->getNewFromOriginal(arg_y);
  Value *data_ptr_y = nullptr;
  IRBuilder<> Builder2(call.getParent());
  switch (Mode) {
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient:
      getReverseBuilder(Builder2);
      break;
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      Builder2.SetInsertPoint(BuilderZ.GetInsertBlock(),
                              BuilderZ.GetInsertPoint());
      Builder2.setFastMathFlags(BuilderZ.getFastMathFlags());
      break;
    case DerivativeMode::ReverseModePrimal:
      break;
  }

  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient ||
      Mode == DerivativeMode::ForwardModeSplit) {

    if (cachetype) {
      if (Mode != DerivativeMode::ReverseModeCombined) {
        cacheval = BuilderZ.CreatePHI(cachetype, 0);
      }
      cacheval = gutils->cacheForReverse(
          BuilderZ, cacheval, getIndex(&call, CacheType::Tape));
      if (Mode != DerivativeMode::ForwardModeSplit)
        cacheval = lookup(cacheval, Builder2);
    }

    if (byRef) {
      if (cache_n) {
        len_n = (cacheTypes.size() == 1)
                    ? cacheval
                    : Builder2.CreateExtractValue(cacheval, {cacheidx});
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(len_n, alloc);
        len_n = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
        cacheidx++;
      } else {
        if (Mode != DerivativeMode::ForwardModeSplit)
          len_n = lookup(len_n, Builder2);
      }

      if (cache_incx) {
        true_incx =
            (cacheTypes.size() == 1)
                ? cacheval
                : Builder2.CreateExtractValue(cacheval, {cacheidx});
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(true_incx, alloc);
        true_incx = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
        incx = true_incx;
        cacheidx++;
      } else if (need_incx) {
        if (Mode != DerivativeMode::ForwardModeSplit) {
          true_incx = lookup(true_incx, Builder2);
          incx = true_incx;
        }
      }

      if (cache_incy) {
        true_incy =
            (cacheTypes.size() == 1)
                ? cacheval
                : Builder2.CreateExtractValue(cacheval, {cacheidx});
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(true_incy, alloc);
        true_incy = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
        incy = true_incy;
        cacheidx++;
      } else if (need_incy) {
        if (Mode != DerivativeMode::ForwardModeSplit) {
          true_incy = lookup(true_incy, Builder2);
          incy = true_incy;
        }
      }

    } else if (Mode != DerivativeMode::ForwardModeSplit) {
      len_n = lookup(len_n, Builder2);

      if (cache_incx) {
        true_incx = lookup(true_incx, Builder2);
        incx = true_incx;
      }
      if (cache_incy) {
        true_incy = lookup(true_incy, Builder2);
        incy = true_incy;
      }
    }
    if (cache_x) {
      data_ptr_x = data_x =
          (cacheTypes.size() == 1)
              ? cacheval
              : Builder2.CreateExtractValue(cacheval, {cacheidx});
      cacheidx++;
      incx = ConstantInt::get(intType, 1);
      if (byRef) {
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(incx, alloc);
        incx = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
      }
      if (type_x->isIntegerTy())
        data_x = Builder2.CreatePtrToInt(data_x, type_x);
    }   else if (active_alpha) {
      data_x = lookup(gutils->getNewFromOriginal(arg_x), Builder2);
    }
    if (cache_y) {
      data_ptr_y = data_y =
          (cacheTypes.size() == 1)
              ? cacheval
              : Builder2.CreateExtractValue(cacheval, {cacheidx});
      cacheidx++;
      incy = ConstantInt::get(intType, 1);
      if (byRef) {
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(incy, alloc);
        incy = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
      }
      if (type_y->isIntegerTy())
        data_y = Builder2.CreatePtrToInt(data_y, type_y);
    }  } else {

    if (type_x->isIntegerTy())
      data_x = Builder2.CreatePtrToInt(data_x, type_x);
    if (type_y->isIntegerTy())
      data_y = Builder2.CreatePtrToInt(data_y, type_y);
  }
  /* fwd-rewrite */                                 
  if (Mode == DerivativeMode::ForwardMode ||        
      Mode == DerivativeMode::ForwardModeSplit) {   
                                                    
#if LLVM_VERSION_MAJOR >= 11                        
    auto callval = call.getCalledOperand();         
#else                                               
    auto callval = call.getCalledValue();           
#endif                                            

    Value *d_alpha = active_alpha
     ? gutils->invertPointerM(arg_alpha, Builder2)
     : nullptr;
    Value *d_x = active_x
     ? gutils->invertPointerM(arg_x, Builder2)
     : nullptr;
    Value *d_y = active_y
     ? gutils->invertPointerM(arg_y, Builder2)
     : nullptr;
    Value *dres = applyChainRule(
        call.getType(), Builder2,
        [&](Value *d_alpha, Value *d_x, Value *d_y  ) {
      Value *dres = nullptr;
      if(active_alpha) {
        Value *args1[] = {len_n, d_alpha, data_x, incx, data_y, incy};

        auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::None, ValueType::Shadow, cache_x ? ValueType::None : ValueType::Primal, ValueType::None, cache_y ? ValueType::None : ValueType::Primal, ValueType::None}, Builder2, /* lookup */ false);
#if LLVM_VERSION_MAJOR > 7
          dres = Builder2.CreateCall(call.getFunctionType(), callval, args1, Defs);
#else
          dres = Builder2.CreateCall(callval, args1, Defs);
#endif
      }
      if(active_x) {
        Value *args1[] = {len_n, fp_alpha, d_x, true_incx, data_y, incy};

        auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::None, cache_alpha ? ValueType::None : ValueType::Primal, ValueType::Shadow, ValueType::None, cache_y ? ValueType::None : ValueType::Primal, ValueType::None}, Builder2, /* lookup */ false);
#if LLVM_VERSION_MAJOR > 7
        Value *nextCall = Builder2.CreateCall(
          call.getFunctionType(), callval, args1, Defs);
#else
        Value *nextCall = Builder2.CreateCall(callval, args1, Defs);
#endif
        if (dres)
          dres = Builder2.CreateFAdd(dres, nextCall);
        else
          dres = nextCall;
      }
      if(active_y) {
        Value *args1[] = {len_n, fp_alpha, data_x, incx, d_y, true_incy};

        auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::None, cache_alpha ? ValueType::None : ValueType::Primal, cache_x ? ValueType::None : ValueType::Primal, ValueType::None, ValueType::Shadow, ValueType::None}, Builder2, /* lookup */ false);
#if LLVM_VERSION_MAJOR > 7
        Value *nextCall = Builder2.CreateCall(
          call.getFunctionType(), callval, args1, Defs);
#else
        Value *nextCall = Builder2.CreateCall(callval, args1, Defs);
#endif
        if (dres)
          dres = Builder2.CreateFAdd(dres, nextCall);
        else
          dres = nextCall;
      }
      return dres;
    },
    d_alpha, d_x, d_y);
    setDiffe(&call, dres, Builder2);
  }
  /* rev-rewrite */                                 
  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient) {
    Value *dif = diffe(&call, Builder2);
    Value *alloc = nullptr;
    if (byRef) {
      alloc = allocationBuilder.CreateAlloca(fpType);
    }

    auto derivcall_dot = gutils->oldFunc->getParent()->getOrInsertFunction(
      (blas.prefix + "dot" + blas.suffix).str(), Builder2.getVoidTy(),
type_n, type_x, type_incx, byRef ? PointerType::getUnqual(call.getType()) : call.getType()
, type_incy);
#if LLVM_VERSION_MAJOR >= 9
    if (auto F = dyn_cast<Function>(derivcall_dot.getCallee()))
#else
    if (auto F = dyn_cast<Function>(derivcall_dot))
#endif
    {
      F->addFnAttr(Attribute::ArgMemOnly);
      if (byRef) {
        F->addParamAttr(0, Attribute::ReadOnly);
        F->addParamAttr(0, Attribute::NoCapture);
        F->addParamAttr(2, Attribute::ReadOnly);
        F->addParamAttr(2, Attribute::NoCapture);
        F->addParamAttr(4, Attribute::ReadOnly);
        F->addParamAttr(4, Attribute::NoCapture);
      }
    }

    auto derivcall_scal = gutils->oldFunc->getParent()->getOrInsertFunction(
      (blas.prefix + "scal" + blas.suffix).str(), Builder2.getVoidTy(),
type_n, type_alpha, byRef ? PointerType::getUnqual(call.getType()) : call.getType()
, type_incy);
#if LLVM_VERSION_MAJOR >= 9
    if (auto F = dyn_cast<Function>(derivcall_scal.getCallee()))
#else
    if (auto F = dyn_cast<Function>(derivcall_scal))
#endif
    {
      F->addFnAttr(Attribute::ArgMemOnly);
      if (byRef) {
        F->addParamAttr(0, Attribute::ReadOnly);
        F->addParamAttr(0, Attribute::NoCapture);
        F->addParamAttr(3, Attribute::ReadOnly);
        F->addParamAttr(3, Attribute::NoCapture);
      }
    }

    // Vector Mode not handled yet
    Value *d_alpha = active_alpha
     ? lookup(gutils->invertPointerM(arg_alpha, Builder2), Builder2)
     : nullptr;
    Value *d_x = active_x
     ? lookup(gutils->invertPointerM(arg_x, Builder2), Builder2)
     : nullptr;
    Value *d_y = active_y
     ? lookup(gutils->invertPointerM(arg_y, Builder2), Builder2)
     : nullptr;
    applyChainRule(
      Builder2,
      [&](Value *d_alpha, Value *d_x, Value *d_y) {
        if (byRef) {
          Builder2.CreateStore(dif, alloc);
          dif = alloc;
        }
      if (active_alpha) {
        Value *args1[] = {len_n, data_x, incx, dif, incy};
        Builder2.CreateCall(
          (blas.prefix + "dot" + blas.suffix).str(), args1,
          gutils->getInvertedBundles(
            &call,
            {ValueType::None, ValueType::Shadow, cache_x ? ValueType::None : ValueType::Primal, ValueType::None, cache_y ? ValueType::None : ValueType::Primal, ValueType::None},
            Builder2, /* lookup */ true));
      }
      if (active_x) {
        Value *args1[] = {len_n, fp_alpha, dif, incy};
        Builder2.CreateCall(
          (blas.prefix + "scal" + blas.suffix).str(), args1,
          gutils->getInvertedBundles(
            &call,
            {ValueType::None, cache_alpha ? ValueType::None : ValueType::Primal, ValueType::Shadow, ValueType::None, cache_y ? ValueType::None : ValueType::Primal, ValueType::None},
            Builder2, /* lookup */ true));
      }
    },
    d_alpha, d_x, d_y    );
  setDiffe(
    &call,
    Constant::getNullValue(gutils->getShadowType(call.getType())),
    Builder2);
  }
  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient ||
      Mode == DerivativeMode::ForwardModeSplit) {
    if (shouldFree()) {
      if (cache_x) {
        CreateDealloc(Builder2, data_ptr_x);
      }
      if (cache_y) {
        CreateDealloc(Builder2, data_ptr_y);
      }
    }
  }
  if (gutils->knownRecomputeHeuristic.find(&call) !=
    gutils->knownRecomputeHeuristic.end()) {
    if (!gutils->knownRecomputeHeuristic[&call]) {
    gutils->cacheForReverse(BuilderZ, newCall,
     getIndex(&call, CacheType::Self));
    }
  }
  return true;
}


bool handle_dot(BlasInfo blas, llvm::CallInst &call, Function *called,
    const std::map<Argument *, bool> &uncacheable_args, Type *fpType) {
  
  CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
  IRBuilder<> BuilderZ(newCall);
  BuilderZ.setFastMathFlags(getFast());
  IRBuilder<> allocationBuilder(gutils->inversionAllocs);
  allocationBuilder.setFastMathFlags(getFast());
  auto &DL = gutils->oldFunc->getParent()->getDataLayout();
  auto arg_n = call.getArgOperand(0);
  auto type_n = arg_n->getType();
  bool uncacheable_n = uncacheable_args.find(called->getArg(0))->second;

  auto arg_x = call.getArgOperand(1);
  auto type_x = arg_x->getType();
  bool uncacheable_x = uncacheable_args.find(called->getArg(1))->second;
  bool active_x = !gutils->isConstantValue(arg_x);

  auto arg_incx = call.getArgOperand(2);
  auto type_incx = arg_incx->getType();
  bool uncacheable_incx = uncacheable_args.find(called->getArg(2))->second;

  auto arg_y = call.getArgOperand(3);
  auto type_y = arg_y->getType();
  bool uncacheable_y = uncacheable_args.find(called->getArg(3))->second;
  bool active_y = !gutils->isConstantValue(arg_y);

  auto arg_incy = call.getArgOperand(4);
  auto type_incy = arg_incy->getType();
  bool uncacheable_incy = uncacheable_args.find(called->getArg(4))->second;

  /* beginning castvalls */
  Type *castvals[2];
  if (auto PT = dyn_cast<PointerType>(type_x))
    castvals[0] = PT;
  else
    castvals[0] = PointerType::getUnqual(fpType);
  if (auto PT = dyn_cast<PointerType>(type_y))
    castvals[1] = PT;
  else
    castvals[1] = PointerType::getUnqual(fpType);
  Value *cacheval;

  /* ending castvalls */
  IntegerType *intType = dyn_cast<IntegerType>(type_n);
  bool byRef = false;
  if (!intType) {
    auto PT = cast<PointerType>(type_n);
    if (blas.suffix.contains("64"))
      intType = IntegerType::get(PT->getContext(), 64);
    else
      intType = IntegerType::get(PT->getContext(), 32);
    byRef = true;
  }

  SmallVector<Type *, 2> cacheTypes;

  // len, fp must be preserved if overwritten
  bool cache_n = false;
  if (byRef && uncacheable_n) {
    cacheTypes.push_back(intType);
    cache_n = true;
  }
  bool cache_x  = Mode != DerivativeMode::ForwardMode &&
          uncacheable_x && active_x && active_y;
  bool cache_incx = false;
  bool need_incx = (active_x  || (!cache_x && (active_x || active_y)));
  if (byRef && uncacheable_incx && need_incx) {
    cacheTypes.push_back(intType);
    cache_incx = true;
   }

  bool cache_y  = Mode != DerivativeMode::ForwardMode &&
          uncacheable_y && active_x && active_y;
  bool cache_incy = false;
  bool need_incy = (active_y  || (!cache_y && (active_x || active_y)));
  if (byRef && uncacheable_incy && need_incy) {
    cacheTypes.push_back(intType);
    cache_incy = true;
   }

  int numCached = (int) cache_x + (int) cache_y;
  if (cache_x)
    cacheTypes.push_back(castvals[0]);
  if (cache_y)
    cacheTypes.push_back(castvals[1]);
  Type *cachetype = nullptr;
  switch (cacheTypes.size()) {
  case 0:
    break;
  case 1:
    cachetype = cacheTypes[0];
    break;
  default:
    cachetype = StructType::get(call.getContext(), cacheTypes);
    break;
  }

  if ((Mode == DerivativeMode::ReverseModeCombined ||
       Mode == DerivativeMode::ReverseModePrimal) && cachetype) {
    SmallVector<Value *, 2> cacheValues;
    Value *len_n = gutils->getNewFromOriginal(arg_n);
    if (byRef) {
      len_n = BuilderZ.CreatePointerCast(len_n, PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
      len_n = BuilderZ.CreateLoad(intType, len_n);
#else
      len_n = BuilderZ.CreateLoad(len_n);
#endif
      if (cache_n)
        cacheValues.push_back(len_n);
    }
    Value *incx = gutils->getNewFromOriginal(arg_incx);
    if (byRef) {
      incx = BuilderZ.CreatePointerCast(incx, PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
      incx = BuilderZ.CreateLoad(intType, incx);
#else
      incx = BuilderZ.CreateLoad(incx);
#endif
      if (cache_incx)
        cacheValues.push_back(incx);
    }
    Value *incy = gutils->getNewFromOriginal(arg_incy);
    if (byRef) {
      incy = BuilderZ.CreatePointerCast(incy, PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
      incy = BuilderZ.CreateLoad(intType, incy);
#else
      incy = BuilderZ.CreateLoad(incy);
#endif
      if (cache_incy)
        cacheValues.push_back(incy);
    }
    if (cache_x) {
      auto dmemcpy = getOrInsertMemcpyStrided(
          *gutils->oldFunc->getParent(), cast<PointerType>(castvals[0]),
          type_n, 0, 0);
      auto malins = CreateAllocation(BuilderZ, fpType, len_n);
      Value *arg = BuilderZ.CreateBitCast(malins, castvals[0]);
      Value *args[4] = {arg,
                         gutils->getNewFromOriginal(arg_x),
                         len_n, incx};
      if (args[1]->getType()->isIntegerTy())
        args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[0]);
      BuilderZ.CreateCall(dmemcpy, args,
          gutils->getInvertedBundles(&call,{
ValueType::None, ValueType::Shadow, ValueType::None, ValueType::None, ValueType::None},
          BuilderZ, /*lookup*/ false));
      cacheValues.push_back(arg);
    }
    if (cache_y) {
      auto dmemcpy = getOrInsertMemcpyStrided(
          *gutils->oldFunc->getParent(), cast<PointerType>(castvals[1]),
          type_n, 0, 0);
      auto malins = CreateAllocation(BuilderZ, fpType, len_n);
      Value *arg = BuilderZ.CreateBitCast(malins, castvals[1]);
      Value *args[4] = {arg,
                         gutils->getNewFromOriginal(arg_y),
                         len_n, incy};
      if (args[1]->getType()->isIntegerTy())
        args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[1]);
      BuilderZ.CreateCall(dmemcpy, args,
          gutils->getInvertedBundles(&call,{
ValueType::None, ValueType::None, ValueType::None, ValueType::Shadow, ValueType::None},
          BuilderZ, /*lookup*/ false));
      cacheValues.push_back(arg);
    }
    if (cacheValues.size() == 1) {
      cacheval = cacheValues[0];
    } else {
      cacheval = UndefValue::get(cachetype);
      for (auto tup : llvm::enumerate(cacheValues))
        cacheval = BuilderZ.CreateInsertValue(cacheval, tup.value(), tup.index());
    }
    gutils->cacheForReverse(BuilderZ, cacheval,
                            getIndex(&call, CacheType::Tape));
  }
  unsigned cacheidx = 0;
  Value *len_n = gutils->getNewFromOriginal(arg_n);
  Value *true_incx = gutils->getNewFromOriginal(arg_incx);
  Value *incx = true_incx;
  Value *data_x = gutils->getNewFromOriginal(arg_x);
  Value *data_ptr_x = nullptr;
  Value *true_incy = gutils->getNewFromOriginal(arg_incy);
  Value *incy = true_incy;
  Value *data_y = gutils->getNewFromOriginal(arg_y);
  Value *data_ptr_y = nullptr;
  IRBuilder<> Builder2(call.getParent());
  switch (Mode) {
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient:
      getReverseBuilder(Builder2);
      break;
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      Builder2.SetInsertPoint(BuilderZ.GetInsertBlock(),
                              BuilderZ.GetInsertPoint());
      Builder2.setFastMathFlags(BuilderZ.getFastMathFlags());
      break;
    case DerivativeMode::ReverseModePrimal:
      break;
  }

  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient ||
      Mode == DerivativeMode::ForwardModeSplit) {

    if (cachetype) {
      if (Mode != DerivativeMode::ReverseModeCombined) {
        cacheval = BuilderZ.CreatePHI(cachetype, 0);
      }
      cacheval = gutils->cacheForReverse(
          BuilderZ, cacheval, getIndex(&call, CacheType::Tape));
      if (Mode != DerivativeMode::ForwardModeSplit)
        cacheval = lookup(cacheval, Builder2);
    }

    if (byRef) {
      if (cache_n) {
        len_n = (cacheTypes.size() == 1)
                    ? cacheval
                    : Builder2.CreateExtractValue(cacheval, {cacheidx});
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(len_n, alloc);
        len_n = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
        cacheidx++;
      } else {
        if (Mode != DerivativeMode::ForwardModeSplit)
          len_n = lookup(len_n, Builder2);
      }

      if (cache_incx) {
        true_incx =
            (cacheTypes.size() == 1)
                ? cacheval
                : Builder2.CreateExtractValue(cacheval, {cacheidx});
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(true_incx, alloc);
        true_incx = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
        incx = true_incx;
        cacheidx++;
      } else if (need_incx) {
        if (Mode != DerivativeMode::ForwardModeSplit) {
          true_incx = lookup(true_incx, Builder2);
          incx = true_incx;
        }
      }

      if (cache_incy) {
        true_incy =
            (cacheTypes.size() == 1)
                ? cacheval
                : Builder2.CreateExtractValue(cacheval, {cacheidx});
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(true_incy, alloc);
        true_incy = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
        incy = true_incy;
        cacheidx++;
      } else if (need_incy) {
        if (Mode != DerivativeMode::ForwardModeSplit) {
          true_incy = lookup(true_incy, Builder2);
          incy = true_incy;
        }
      }

    } else if (Mode != DerivativeMode::ForwardModeSplit) {
      len_n = lookup(len_n, Builder2);

      if (cache_incx) {
        true_incx = lookup(true_incx, Builder2);
        incx = true_incx;
      }
      if (cache_incy) {
        true_incy = lookup(true_incy, Builder2);
        incy = true_incy;
      }
    }
    if (cache_x) {
      data_ptr_x = data_x =
          (cacheTypes.size() == 1)
              ? cacheval
              : Builder2.CreateExtractValue(cacheval, {cacheidx});
      cacheidx++;
      incx = ConstantInt::get(intType, 1);
      if (byRef) {
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(incx, alloc);
        incx = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
      }
      if (type_x->isIntegerTy())
        data_x = Builder2.CreatePtrToInt(data_x, type_x);
    }   else if (active_x || active_y) {
      data_x = lookup(gutils->getNewFromOriginal(arg_x), Builder2);
    }
    if (cache_y) {
      data_ptr_y = data_y =
          (cacheTypes.size() == 1)
              ? cacheval
              : Builder2.CreateExtractValue(cacheval, {cacheidx});
      cacheidx++;
      incy = ConstantInt::get(intType, 1);
      if (byRef) {
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(incy, alloc);
        incy = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
      }
      if (type_y->isIntegerTy())
        data_y = Builder2.CreatePtrToInt(data_y, type_y);
    }   else if (active_x || active_y) {
      data_y = lookup(gutils->getNewFromOriginal(arg_y), Builder2);
    }
  } else {

    if (type_x->isIntegerTy())
      data_x = Builder2.CreatePtrToInt(data_x, type_x);
    if (type_y->isIntegerTy())
      data_y = Builder2.CreatePtrToInt(data_y, type_y);
  }
  /* fwd-rewrite */                                 
  if (Mode == DerivativeMode::ForwardMode ||        
      Mode == DerivativeMode::ForwardModeSplit) {   
                                                    
#if LLVM_VERSION_MAJOR >= 11                        
    auto callval = call.getCalledOperand();         
#else                                               
    auto callval = call.getCalledValue();           
#endif                                            

    Value *d_x = active_x
     ? gutils->invertPointerM(arg_x, Builder2)
     : nullptr;
    Value *d_y = active_y
     ? gutils->invertPointerM(arg_y, Builder2)
     : nullptr;
    Value *dres = applyChainRule(
        call.getType(), Builder2,
        [&](Value *d_x, Value *d_y  ) {
      Value *dres = nullptr;
      if(active_x) {
        Value *args1[] = {len_n, d_x, true_incx, data_y, incy};

        auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::None, ValueType::Shadow, ValueType::None, cache_y ? ValueType::None : ValueType::Primal, ValueType::None}, Builder2, /* lookup */ false);
#if LLVM_VERSION_MAJOR > 7
          dres = Builder2.CreateCall(call.getFunctionType(), callval, args1, Defs);
#else
          dres = Builder2.CreateCall(callval, args1, Defs);
#endif
      }
      if(active_y) {
        Value *args1[] = {len_n, data_x, incx, d_y, true_incy};

        auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::None, cache_x ? ValueType::None : ValueType::Primal, ValueType::None, ValueType::Shadow, ValueType::None}, Builder2, /* lookup */ false);
#if LLVM_VERSION_MAJOR > 7
        Value *nextCall = Builder2.CreateCall(
          call.getFunctionType(), callval, args1, Defs);
#else
        Value *nextCall = Builder2.CreateCall(callval, args1, Defs);
#endif
        if (dres)
          dres = Builder2.CreateFAdd(dres, nextCall);
        else
          dres = nextCall;
      }
      return dres;
    },
    d_x, d_y);
    setDiffe(&call, dres, Builder2);
  }
  /* rev-rewrite */                                 
  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient) {
    Value *dif = diffe(&call, Builder2);
    Value *alloc = nullptr;
    if (byRef) {
      alloc = allocationBuilder.CreateAlloca(fpType);
    }

    auto derivcall_axpy = gutils->oldFunc->getParent()->getOrInsertFunction(
      (blas.prefix + "axpy" + blas.suffix).str(), Builder2.getVoidTy(),
type_n, byRef ? PointerType::getUnqual(call.getType()) : call.getType()
, type_y, type_incy, type_x, type_incx);
#if LLVM_VERSION_MAJOR >= 9
    if (auto F = dyn_cast<Function>(derivcall_axpy.getCallee()))
#else
    if (auto F = dyn_cast<Function>(derivcall_axpy))
#endif
    {
      F->addFnAttr(Attribute::ArgMemOnly);
      if (byRef) {
        F->addParamAttr(0, Attribute::ReadOnly);
        F->addParamAttr(0, Attribute::NoCapture);
        F->addParamAttr(3, Attribute::ReadOnly);
        F->addParamAttr(3, Attribute::NoCapture);
        F->addParamAttr(5, Attribute::ReadOnly);
        F->addParamAttr(5, Attribute::NoCapture);
      }
    }

    // Vector Mode not handled yet
    Value *d_x = active_x
     ? lookup(gutils->invertPointerM(arg_x, Builder2), Builder2)
     : nullptr;
    Value *d_y = active_y
     ? lookup(gutils->invertPointerM(arg_y, Builder2), Builder2)
     : nullptr;
    applyChainRule(
      Builder2,
      [&](Value *d_x, Value *d_y) {
        if (byRef) {
          Builder2.CreateStore(dif, alloc);
          dif = alloc;
        }
      if (active_x) {
        Value *args1[] = {len_n, dif, data_y, incy, d_x, true_incx};
        Builder2.CreateCall(
          (blas.prefix + "axpy" + blas.suffix).str(), args1,
          gutils->getInvertedBundles(
            &call,
            {ValueType::None, ValueType::Shadow, ValueType::None, cache_y ? ValueType::None : ValueType::Primal, ValueType::None},
            Builder2, /* lookup */ true));
      }
      if (active_y) {
        Value *args1[] = {len_n, dif, data_x, incx, d_y, true_incy};
        Builder2.CreateCall(
          (blas.prefix + "axpy" + blas.suffix).str(), args1,
          gutils->getInvertedBundles(
            &call,
            {ValueType::None, cache_x ? ValueType::None : ValueType::Primal, ValueType::None, ValueType::Shadow, ValueType::None},
            Builder2, /* lookup */ true));
      }
    },
    d_x, d_y    );
  setDiffe(
    &call,
    Constant::getNullValue(gutils->getShadowType(call.getType())),
    Builder2);
  }
  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient ||
      Mode == DerivativeMode::ForwardModeSplit) {
    if (shouldFree()) {
      if (cache_x) {
        CreateDealloc(Builder2, data_ptr_x);
      }
      if (cache_y) {
        CreateDealloc(Builder2, data_ptr_y);
      }
    }
  }
  if (gutils->knownRecomputeHeuristic.find(&call) !=
    gutils->knownRecomputeHeuristic.end()) {
    if (!gutils->knownRecomputeHeuristic[&call]) {
    gutils->cacheForReverse(BuilderZ, newCall,
     getIndex(&call, CacheType::Self));
    }
  }
  return true;
}


bool handle_scal(BlasInfo blas, llvm::CallInst &call, Function *called,
    const std::map<Argument *, bool> &uncacheable_args, Type *fpType) {
  
  CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
  IRBuilder<> BuilderZ(newCall);
  BuilderZ.setFastMathFlags(getFast());
  IRBuilder<> allocationBuilder(gutils->inversionAllocs);
  allocationBuilder.setFastMathFlags(getFast());
  auto &DL = gutils->oldFunc->getParent()->getDataLayout();
  auto arg_n = call.getArgOperand(0);
  auto type_n = arg_n->getType();
  bool uncacheable_n = uncacheable_args.find(called->getArg(0))->second;

  auto arg_alpha = call.getArgOperand(1);
  auto type_alpha = arg_alpha->getType();
  bool uncacheable_alpha = uncacheable_args.find(called->getArg(1))->second;
  bool active_alpha = !gutils->isConstantValue(arg_alpha);

  auto arg_x = call.getArgOperand(2);
  auto type_x = arg_x->getType();
  bool uncacheable_x = uncacheable_args.find(called->getArg(2))->second;
  bool active_x = !gutils->isConstantValue(arg_x);

  auto arg_incx = call.getArgOperand(3);
  auto type_incx = arg_incx->getType();
  bool uncacheable_incx = uncacheable_args.find(called->getArg(3))->second;

  /* beginning castvalls */
  Type *castvals[2];
  if (auto PT = dyn_cast<PointerType>(type_alpha))
    castvals[0] = PT;
  else
    castvals[0] = PointerType::getUnqual(fpType);
  if (auto PT = dyn_cast<PointerType>(type_x))
    castvals[1] = PT;
  else
    castvals[1] = PointerType::getUnqual(fpType);
  Value *cacheval;

  /* ending castvalls */
  IntegerType *intType = dyn_cast<IntegerType>(type_n);
  bool byRef = false;
  if (!intType) {
    auto PT = cast<PointerType>(type_n);
    if (blas.suffix.contains("64"))
      intType = IntegerType::get(PT->getContext(), 64);
    else
      intType = IntegerType::get(PT->getContext(), 32);
    byRef = true;
  }

  SmallVector<Type *, 2> cacheTypes;

  // len, fp must be preserved if overwritten
  bool cache_n = false;
  if (byRef && uncacheable_n) {
    cacheTypes.push_back(intType);
    cache_n = true;
  }
  bool cache_alpha = false;
  if (byRef && uncacheable_alpha) {
    cacheTypes.push_back(fpType);
    cache_alpha = true;
  }
  bool cache_x  = Mode != DerivativeMode::ForwardMode &&
          uncacheable_x && active_alpha && active_x;
  bool cache_incx = false;
  bool need_incx = (active_x  || (!cache_x && (active_alpha || active_x)));
  if (byRef && uncacheable_incx && need_incx) {
    cacheTypes.push_back(intType);
    cache_incx = true;
   }

  int numCached = (int) cache_x;
  if (cache_alpha)
    cacheTypes.push_back(castvals[0]);
  if (cache_x)
    cacheTypes.push_back(castvals[1]);
  Type *cachetype = nullptr;
  switch (cacheTypes.size()) {
  case 0:
    break;
  case 1:
    cachetype = cacheTypes[0];
    break;
  default:
    cachetype = StructType::get(call.getContext(), cacheTypes);
    break;
  }

  if ((Mode == DerivativeMode::ReverseModeCombined ||
       Mode == DerivativeMode::ReverseModePrimal) && cachetype) {
    SmallVector<Value *, 2> cacheValues;
    Value *len_n = gutils->getNewFromOriginal(arg_n);
    if (byRef) {
      len_n = BuilderZ.CreatePointerCast(len_n, PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
      len_n = BuilderZ.CreateLoad(intType, len_n);
#else
      len_n = BuilderZ.CreateLoad(len_n);
#endif
      if (cache_n)
        cacheValues.push_back(len_n);
    }
    Value *fp_alpha = gutils->getNewFromOriginal(arg_alpha);
    if (byRef) {
      fp_alpha = BuilderZ.CreatePointerCast(fp_alpha, PointerType::getUnqual(fpType));
#if LLVM_VERSION_MAJOR > 7
      fp_alpha = BuilderZ.CreateLoad(fpType, fp_alpha);
#else
      fp_alpha = BuilderZ.CreateLoad(fp_alpha);
#endif
      if (cache_alpha)
        cacheValues.push_back(fp_alpha);
    }
    Value *incx = gutils->getNewFromOriginal(arg_incx);
    if (byRef) {
      incx = BuilderZ.CreatePointerCast(incx, PointerType::getUnqual(intType));
#if LLVM_VERSION_MAJOR > 7
      incx = BuilderZ.CreateLoad(intType, incx);
#else
      incx = BuilderZ.CreateLoad(incx);
#endif
      if (cache_incx)
        cacheValues.push_back(incx);
    }
    if (cache_x) {
      auto dmemcpy = getOrInsertMemcpyStrided(
          *gutils->oldFunc->getParent(), cast<PointerType>(castvals[0]),
          type_n, 0, 0);
      auto malins = CreateAllocation(BuilderZ, fpType, len_n);
      Value *arg = BuilderZ.CreateBitCast(malins, castvals[0]);
      Value *args[4] = {arg,
                         gutils->getNewFromOriginal(arg_x),
                         len_n, incx};
      if (args[1]->getType()->isIntegerTy())
        args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[0]);
      BuilderZ.CreateCall(dmemcpy, args,
          gutils->getInvertedBundles(&call,{
ValueType::None, ValueType::None, ValueType::Shadow, ValueType::None},
          BuilderZ, /*lookup*/ false));
      cacheValues.push_back(arg);
    }
    if (cacheValues.size() == 1) {
      cacheval = cacheValues[0];
    } else {
      cacheval = UndefValue::get(cachetype);
      for (auto tup : llvm::enumerate(cacheValues))
        cacheval = BuilderZ.CreateInsertValue(cacheval, tup.value(), tup.index());
    }
    gutils->cacheForReverse(BuilderZ, cacheval,
                            getIndex(&call, CacheType::Tape));
  }
  unsigned cacheidx = 0;
  Value *len_n = gutils->getNewFromOriginal(arg_n);
  Value *fp_alpha = gutils->getNewFromOriginal(arg_alpha);
  Value *true_incx = gutils->getNewFromOriginal(arg_incx);
  Value *incx = true_incx;
  Value *data_x = gutils->getNewFromOriginal(arg_x);
  Value *data_ptr_x = nullptr;
  IRBuilder<> Builder2(call.getParent());
  switch (Mode) {
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient:
      getReverseBuilder(Builder2);
      break;
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      Builder2.SetInsertPoint(BuilderZ.GetInsertBlock(),
                              BuilderZ.GetInsertPoint());
      Builder2.setFastMathFlags(BuilderZ.getFastMathFlags());
      break;
    case DerivativeMode::ReverseModePrimal:
      break;
  }

  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient ||
      Mode == DerivativeMode::ForwardModeSplit) {

    if (cachetype) {
      if (Mode != DerivativeMode::ReverseModeCombined) {
        cacheval = BuilderZ.CreatePHI(cachetype, 0);
      }
      cacheval = gutils->cacheForReverse(
          BuilderZ, cacheval, getIndex(&call, CacheType::Tape));
      if (Mode != DerivativeMode::ForwardModeSplit)
        cacheval = lookup(cacheval, Builder2);
    }

    if (byRef) {
      if (cache_n) {
        len_n = (cacheTypes.size() == 1)
                    ? cacheval
                    : Builder2.CreateExtractValue(cacheval, {cacheidx});
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(len_n, alloc);
        len_n = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
        cacheidx++;
      } else {
        if (Mode != DerivativeMode::ForwardModeSplit)
          len_n = lookup(len_n, Builder2);
      }

      if (cache_incx) {
        true_incx =
            (cacheTypes.size() == 1)
                ? cacheval
                : Builder2.CreateExtractValue(cacheval, {cacheidx});
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(true_incx, alloc);
        true_incx = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
        incx = true_incx;
        cacheidx++;
      } else if (need_incx) {
        if (Mode != DerivativeMode::ForwardModeSplit) {
          true_incx = lookup(true_incx, Builder2);
          incx = true_incx;
        }
      }

    } else if (Mode != DerivativeMode::ForwardModeSplit) {
      len_n = lookup(len_n, Builder2);

      if (cache_incx) {
        true_incx = lookup(true_incx, Builder2);
        incx = true_incx;
      }
    }
    if (cache_x) {
      data_ptr_x = data_x =
          (cacheTypes.size() == 1)
              ? cacheval
              : Builder2.CreateExtractValue(cacheval, {cacheidx});
      cacheidx++;
      incx = ConstantInt::get(intType, 1);
      if (byRef) {
        auto alloc = allocationBuilder.CreateAlloca(intType);
        Builder2.CreateStore(incx, alloc);
        incx = Builder2.CreatePointerCast(
            alloc, call.getArgOperand(0)->getType());
      }
      if (type_x->isIntegerTy())
        data_x = Builder2.CreatePtrToInt(data_x, type_x);
    }   else if (active_alpha || active_x) {
      data_x = lookup(gutils->getNewFromOriginal(arg_x), Builder2);
    }
  } else {

    if (type_x->isIntegerTy())
      data_x = Builder2.CreatePtrToInt(data_x, type_x);
  }
  /* fwd-rewrite */                                 
  if (Mode == DerivativeMode::ForwardMode ||        
      Mode == DerivativeMode::ForwardModeSplit) {   
                                                    
#if LLVM_VERSION_MAJOR >= 11                        
    auto callval = call.getCalledOperand();         
#else                                               
    auto callval = call.getCalledValue();           
#endif                                            

    Value *d_alpha = active_alpha
     ? gutils->invertPointerM(arg_alpha, Builder2)
     : nullptr;
    Value *d_x = active_x
     ? gutils->invertPointerM(arg_x, Builder2)
     : nullptr;
    Value *dres = applyChainRule(
        call.getType(), Builder2,
        [&](Value *d_alpha, Value *d_x  ) {
      Value *dres = nullptr;
      if(active_alpha) {
        Value *args1[] = {len_n, d_alpha, data_x, incx};

        auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::None, ValueType::Shadow, cache_x ? ValueType::None : ValueType::Primal, ValueType::None}, Builder2, /* lookup */ false);
#if LLVM_VERSION_MAJOR > 7
          dres = Builder2.CreateCall(call.getFunctionType(), callval, args1, Defs);
#else
          dres = Builder2.CreateCall(callval, args1, Defs);
#endif
      }
      if(active_x) {
        Value *args1[] = {len_n, fp_alpha, d_x, true_incx};

        auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::None, cache_alpha ? ValueType::None : ValueType::Primal, ValueType::Shadow, ValueType::None}, Builder2, /* lookup */ false);
#if LLVM_VERSION_MAJOR > 7
        Value *nextCall = Builder2.CreateCall(
          call.getFunctionType(), callval, args1, Defs);
#else
        Value *nextCall = Builder2.CreateCall(callval, args1, Defs);
#endif
        if (dres)
          dres = Builder2.CreateFAdd(dres, nextCall);
        else
          dres = nextCall;
      }
      return dres;
    },
    d_alpha, d_x);
    setDiffe(&call, dres, Builder2);
  }
  /* rev-rewrite */                                 
  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient) {
    Value *dif = diffe(&call, Builder2);
    Value *alloc = nullptr;
    if (byRef) {
      alloc = allocationBuilder.CreateAlloca(fpType);
    }

    auto derivcall_dot = gutils->oldFunc->getParent()->getOrInsertFunction(
      (blas.prefix + "dot" + blas.suffix).str(), Builder2.getVoidTy(),
type_n, type_x, type_incx, byRef ? PointerType::getUnqual(call.getType()) : call.getType()
, type_incx);
#if LLVM_VERSION_MAJOR >= 9
    if (auto F = dyn_cast<Function>(derivcall_dot.getCallee()))
#else
    if (auto F = dyn_cast<Function>(derivcall_dot))
#endif
    {
      F->addFnAttr(Attribute::ArgMemOnly);
      if (byRef) {
        F->addParamAttr(0, Attribute::ReadOnly);
        F->addParamAttr(0, Attribute::NoCapture);
        F->addParamAttr(2, Attribute::ReadOnly);
        F->addParamAttr(2, Attribute::NoCapture);
        F->addParamAttr(4, Attribute::ReadOnly);
        F->addParamAttr(4, Attribute::NoCapture);
      }
    }

    auto derivcall_axpy = gutils->oldFunc->getParent()->getOrInsertFunction(
      (blas.prefix + "axpy" + blas.suffix).str(), Builder2.getVoidTy(),
type_n, type_alpha, byRef ? PointerType::getUnqual(call.getType()) : call.getType()
, type_incx, type_x, type_incx);
#if LLVM_VERSION_MAJOR >= 9
    if (auto F = dyn_cast<Function>(derivcall_axpy.getCallee()))
#else
    if (auto F = dyn_cast<Function>(derivcall_axpy))
#endif
    {
      F->addFnAttr(Attribute::ArgMemOnly);
      if (byRef) {
        F->addParamAttr(0, Attribute::ReadOnly);
        F->addParamAttr(0, Attribute::NoCapture);
        F->addParamAttr(3, Attribute::ReadOnly);
        F->addParamAttr(3, Attribute::NoCapture);
        F->addParamAttr(5, Attribute::ReadOnly);
        F->addParamAttr(5, Attribute::NoCapture);
      }
    }

    // Vector Mode not handled yet
    Value *d_alpha = active_alpha
     ? lookup(gutils->invertPointerM(arg_alpha, Builder2), Builder2)
     : nullptr;
    Value *d_x = active_x
     ? lookup(gutils->invertPointerM(arg_x, Builder2), Builder2)
     : nullptr;
    applyChainRule(
      Builder2,
      [&](Value *d_alpha, Value *d_x) {
        if (byRef) {
          Builder2.CreateStore(dif, alloc);
          dif = alloc;
        }
      if (active_alpha) {
        Value *args1[] = {len_n, data_x, incx, dif, incx};
        Builder2.CreateCall(
          (blas.prefix + "dot" + blas.suffix).str(), args1,
          gutils->getInvertedBundles(
            &call,
            {ValueType::None, ValueType::Shadow, cache_x ? ValueType::None : ValueType::Primal, ValueType::None},
            Builder2, /* lookup */ true));
      }
      if (active_x) {
        Value *args1[] = {len_n, fp_alpha, dif, incx, d_x, true_incx};
        Builder2.CreateCall(
          (blas.prefix + "axpy" + blas.suffix).str(), args1,
          gutils->getInvertedBundles(
            &call,
            {ValueType::None, cache_alpha ? ValueType::None : ValueType::Primal, ValueType::Shadow, ValueType::None},
            Builder2, /* lookup */ true));
      }
    },
    d_alpha, d_x    );
  setDiffe(
    &call,
    Constant::getNullValue(gutils->getShadowType(call.getType())),
    Builder2);
  }
  if (Mode == DerivativeMode::ReverseModeCombined ||
      Mode == DerivativeMode::ReverseModeGradient ||
      Mode == DerivativeMode::ForwardModeSplit) {
    if (shouldFree()) {
      if (cache_x) {
        CreateDealloc(Builder2, data_ptr_x);
      }
    }
  }
  if (gutils->knownRecomputeHeuristic.find(&call) !=
    gutils->knownRecomputeHeuristic.end()) {
    if (!gutils->knownRecomputeHeuristic[&call]) {
    gutils->cacheForReverse(BuilderZ, newCall,
     getIndex(&call, CacheType::Self));
    }
  }
  return true;
}

