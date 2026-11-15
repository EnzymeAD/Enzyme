# The !DIR$ ENZYME directive tests. lit gates them on the flang_directives
# feature: a flang with plugin directives, and FlangEnzymeMLIR built for it.
set(ENZYME_DIRECTIVES_TEST_DEPS ${ENZYME_TEST_DEPS})
if(TARGET FlangEnzymeMLIR-${LLVM_VERSION_MAJOR})
  list(APPEND ENZYME_DIRECTIVES_TEST_DEPS FlangEnzymeMLIR-${LLVM_VERSION_MAJOR})
endif()
if(TARGET FlangEnzyme-${LLVM_VERSION_MAJOR})
  list(APPEND ENZYME_DIRECTIVES_TEST_DEPS FlangEnzyme-${LLVM_VERSION_MAJOR})
endif()

add_lit_testsuite(
  check-enzyme-fortran-directives
  "Running enzyme !DIR$ ENZYME directive tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${ENZYME_DIRECTIVES_TEST_DEPS}
  ARGS
  -v)

set_target_properties(check-enzyme-fortran-directives PROPERTIES FOLDER "Tests")
