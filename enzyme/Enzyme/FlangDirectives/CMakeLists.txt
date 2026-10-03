# FlangEnzymeDirectives: a `flang -fc1 -load` plugin that defines the
# !DIR$ ENZYME directives and turns them, and the Fortran types LLVM IR
# erases, into what LLVM Enzyme reads. It needs a flang with plugin
# directives and pass pipeline callbacks, and MLIR and flang's CMake packages
# (-DMLIR_DIR=<llvm>/lib/cmake/mlir -DFlang_DIR=<llvm>/lib/cmake/flang, with
# -DClang_DIR too, which FlangConfig refers to). It does not need Enzyme-MLIR.
if(NOT MLIR_FOUND)
  if(NOT DEFINED MLIR_DIR)
    set(MLIR_DIR ${LLVM_DIR}/../mlir)
  endif()
  find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR} NO_DEFAULT_PATH)
endif()
if(NOT Flang_FOUND)
  if(NOT DEFINED Clang_DIR)
    set(Clang_DIR ${LLVM_DIR}/../clang)
  endif()
  find_package(Clang REQUIRED CONFIG PATHS ${Clang_DIR} NO_DEFAULT_PATH)
  if(NOT DEFINED Flang_DIR)
    set(Flang_DIR ${LLVM_DIR}/../flang)
  endif()
  find_package(Flang REQUIRED CONFIG PATHS ${Flang_DIR} NO_DEFAULT_PATH)
endif()

add_llvm_library(
  FlangEnzymeDirectives-${LLVM_VERSION_MAJOR} MODULE FlangDirectivesPlugin.cpp
  FortranDirectives.cpp FIRTypeAnnotations.cpp DISABLE_LLVM_LINK_LLVM_DYLIB)
target_include_directories(
  FlangEnzymeDirectives-${LLVM_VERSION_MAJOR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${MLIR_INCLUDE_DIRS} ${FLANG_INCLUDE_DIRS})
# MLIR, FIR and flang symbols resolve from the host `flang -fc1` at load.
target_link_options(
  FlangEnzymeDirectives-${LLVM_VERSION_MAJOR}
  PRIVATE -Wl,--allow-shlib-undefined)

install(
  TARGETS FlangEnzymeDirectives-${LLVM_VERSION_MAJOR}
  LIBRARY DESTINATION lib COMPONENT shlib)
