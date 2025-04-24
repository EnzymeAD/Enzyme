set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(enzymemlir-translate
  enzymemlir-translate.cpp
  )
llvm_update_compile_flags(enzymemlir-translate)
target_link_libraries(enzymemlir-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  "ZLIB::ZLIB"
  )

mlir_check_link_libraries(enzymemlir-translate)
