//
// Created by Chuyang Chen on 14/7/2021.
//

#ifndef ENZYME_RUSTDEBUGINFO_H
#define ENZYME_RUSTDEBUGINFO_H 1

#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;

#include "TypeTree.h"

TypeTree parseDIType(DbgDeclareInst& I, DataLayout& DL);

#endif //ENZYME_RUSTDEBUGINFO_H
