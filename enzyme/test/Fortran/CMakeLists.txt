message("Building Fortran tests")

add_subdirectory(BatchMode)
if(ENZYME_FLANG_DIRECTIVES)
  list(APPEND ENZYME_TEST_DEPS FlangEnzymeDirectives-${LLVM_VERSION_MAJOR})
  add_subdirectory(Directives)
endif()
add_subdirectory(ForwardMode)
add_subdirectory(mpi)
add_subdirectory(OpenMP)
add_subdirectory(ReverseMode)
add_subdirectory(TypeAnalysis)

# Run regression and unit tests
add_lit_testsuite(
  check-enzyme-fortran
  "Running enzyme fortran integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${ENZYME_TEST_DEPS}
  LLVMEnzyme-${LLVM_VERSION_MAJOR}
  ARGS
  -v)

set_target_properties(check-enzyme-fortran PROPERTIES FOLDER "Tests")
