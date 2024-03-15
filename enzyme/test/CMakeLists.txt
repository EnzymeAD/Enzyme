configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ENZYME_TEST_DEPS LLVMEnzyme-${LLVM_VERSION_MAJOR})

add_subdirectory(ActivityAnalysis)
add_subdirectory(TypeAnalysis)
add_subdirectory(Enzyme)
if (${Clang_FOUND})
add_subdirectory(Integration)
if (${LLVM_VERSION_MAJOR} GREATER_EQUAL 11)
add_custom_target(test-cmake COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/test_cmake && mkdir ${CMAKE_CURRENT_BINARY_DIR}/test_cmake && cd ${CMAKE_CURRENT_BINARY_DIR}/test_cmake &&  ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_BINARY_DIR}/test_cmake ${CMAKE_CURRENT_SOURCE_DIR}/test_find_package -DEnzyme_DIR=${CMAKE_BINARY_DIR} -DCMAKE_C_COMPILER=$<TARGET_FILE:clang> && ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/test_cmake DEPENDS ClangEnzymeFlags LLDEnzymeFlags)
else()
add_custom_target(test-cmake COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/test_cmake && mkdir ${CMAKE_CURRENT_BINARY_DIR}/test_cmake && cd ${CMAKE_CURRENT_BINARY_DIR}/test_cmake &&  ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_BINARY_DIR}/test_cmake ${CMAKE_CURRENT_SOURCE_DIR}/test_find_package -DEnzyme_DIR=${CMAKE_BINARY_DIR} -DCMAKE_C_COMPILER=$<TARGET_FILE:clang> && ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/test_cmake DEPENDS ClangEnzymeFlags)
endif()
endif()
if (ENZYME_IFX)
    add_subdirectory(Fortran)
endif()
add_subdirectory(BCLoader)
if (ENZYME_MLIR)
    add_subdirectory(MLIR)
endif()
