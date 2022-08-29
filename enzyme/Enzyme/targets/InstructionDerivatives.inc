// clang-format off
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

  if (funcName == "atan" ||
      funcName == "atanf" ||
      funcName == "atanl" ||
      funcName == "__fd_atan_1" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[2];
 args[0] = ConstantFP::get(orig->getOperand(0)->getType(), "1.0") ;
 args[1] =  ({
    Value* args[2];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[2];
 args[0] = ConstantFP::get(orig->getOperand(0)->getType(), "1.0") ;
 args[1] =  ({
    Value* args[2];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "tanhf" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("coshf", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value* __tmp_c = args[0];
 args[1] = __tmp_c ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("coshf", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value* __tmp_c = args[0];
 args[1] = __tmp_c ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "cosh" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("sinh", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("sinh", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "coshf" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("sinhf", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("sinhf", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "sinh" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("cosh", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("cosh", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "sinhf" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("coshf", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("coshf", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "exp10" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] = ConstantFP::get(orig->getOperand(0)->getType(), "2.30258509299404568401799145468") ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] = ConstantFP::get(orig->getOperand(0)->getType(), "2.30258509299404568401799145468") ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "tan" ||
      funcName == "tanf" ||
      funcName == "tanl" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[2];
 args[0] = ConstantFP::get(orig->getOperand(0)->getType(), "1.0") ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value* __tmp_c = args[0];
 args[1] = __tmp_c ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[2];
 args[0] = ConstantFP::get(orig->getOperand(0)->getType(), "1.0") ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value* __tmp_c = args[0];
 args[1] = __tmp_c ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "expm1" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = ConstantFP::get(orig->getOperand(0)->getType(), "1.0") ;
 args[1] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 args[1] = dif ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(args[0], Builder2.CreateExtractValue(args[1], {idx}));
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = ConstantFP::get(orig->getOperand(0)->getType(), "1.0") ;
 args[1] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 args[1] = dif ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(args[0], Builder2.CreateExtractValue(args[1], {idx}));
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "atan2" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(1)) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(1)) ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(1)) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        if (!gutils->isConstantValue(orig->getArgOperand(1))) {
          Value *dif = diffe(orig->getArgOperand(1), Builder2);
          Value *tmp =  ({
    Value* args[1];
 args[0] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(1)) ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(1)) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFNeg(args[0]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFNeg(Builder2.CreateExtractValue(args[0], {idx}));
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        } else if (!dif && !gutils->isConstantValue(orig->getArgOperand(1))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(1)), Builder2) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(1)), Builder2) ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(1)), Builder2) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        if (!gutils->isConstantValue(orig->getArgOperand(1))) {
          Value *tmp =  ({
    Value* args[1];
 args[0] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(1)), Builder2) ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(1)), Builder2) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
   res = Builder2.CreateFAdd(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFNeg(args[0]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFNeg(Builder2.CreateExtractValue(args[0], {idx}));
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(1), toadd, Builder2, orig->getArgOperand(1)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "cbrt" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = ConstantFP::get(orig->getOperand(0)->getType(), "3.0") ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = ConstantFP::get(orig->getOperand(0)->getType(), "3.0") ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "hypot" ||
      funcName == "hypotf" ||
      funcName == "hypotl" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(1)) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0], args[1]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        if (!gutils->isConstantValue(orig->getArgOperand(1))) {
          Value *dif = diffe(orig->getArgOperand(1), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(1)) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
 args[1] = gutils->getNewFromOriginal(orig->getOperand(1)) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0], args[1]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        } else if (!dif && !gutils->isConstantValue(orig->getArgOperand(1))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(1)), Builder2) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0], args[1]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        if (!gutils->isConstantValue(orig->getArgOperand(1))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(1)), Builder2) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFMul(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFMul(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; }) ;
 args[1] =  ({
    Value* args[2];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
 args[1] = lookup(gutils->getNewFromOriginal(orig->getOperand(1)), Builder2) ;
#if LLVM_VERSION_MAJOR >= 11
  auto callval = orig->getCalledOperand();
#else
  auto callval = orig->getCalledValue();
#endif
  auto FT = orig->getFunctionType();
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0], args[1]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(1), toadd, Builder2, orig->getArgOperand(1)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
  if (funcName == "tanh" ){
    if (gutils->knownRecomputeHeuristic.find(orig) !=
        gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[orig]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(orig, CacheType::Self));
        }
    }
    eraseIfUnused(*orig);
    if (gutils->isConstantInstruction(orig))
      return;
    switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode:{
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        Value *res = nullptr;
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *dif = diffe(orig->getArgOperand(0), Builder2);
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[1];
 args[0] = gutils->getNewFromOriginal(orig->getOperand(0)) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("cosh", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value* __tmp_c = args[0];
 args[1] = __tmp_c ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          if (res == nullptr) res = tmp;
          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);
          else {
            Value *out = UndefValue::get(res->getType());
            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));
              out = Builder2.CreateInsertValue(out, V, {idx});
            }
            res = out;
          }
        }
        setDiffe(orig, res, Builder2);
        break;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined:{
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *dif = nullptr;
        if (!dif && !gutils->isConstantValue(orig->getArgOperand(0))) {
          dif = diffe(orig, Builder2);
          setDiffe(orig, Constant::getNullValue(gutils->getShadowType(orig->getType())), Builder2);
        }
        if (!gutils->isConstantValue(orig->getArgOperand(0))) {
          Value *tmp =  ({
    Value* args[2];
 args[0] = dif ;
 args[1] =  ({
    Value* args[2];
 args[0] =  ({
    Value* args[1];
 args[0] = lookup(gutils->getNewFromOriginal(orig->getOperand(0)), Builder2) ;
  auto FT = orig->getFunctionType();
  auto callval = gutils->oldFunc->getParent()->getOrInsertFunction("cosh", FT, called->getAttributes())
#if LLVM_VERSION_MAJOR >= 9
  .getCallee()
#endif
;
  auto cconv = orig->getCallingConv();
 Value *res = nullptr;
 CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(FT, callval, ArrayRef<Value*>({args[0]})));
 cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));
 cubcall->setCallingConv(cconv);
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
#if LLVM_VERSION_MAJOR >= 14
 cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, Attribute::NoUnwind);
#else
 cubcall->addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
#endif
 res = cubcall;
 res; }) ;
 Value* __tmp_c = args[0];
 args[1] = __tmp_c ;
 Value *res = nullptr;
   res = Builder2.CreateFMul(args[0], args[1]);
 res; }) ;
 Value *res = nullptr;
 if (gutils->getWidth() == 1) { 
   res = Builder2.CreateFDiv(args[0], args[1]);
 } else {
 for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {
   Value *V = Builder2.CreateFDiv(Builder2.CreateExtractValue(args[0], {idx}), args[1]);
   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));
   res = Builder2.CreateInsertValue(res, V, {idx});
 }
 }
 res; });
          Value *toadd = tmp;
          addToDiffe(orig->getArgOperand(0), toadd, Builder2, orig->getArgOperand(0)->getType());
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:{
        break;
      }
    }
    return;
  }
