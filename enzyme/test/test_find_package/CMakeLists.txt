cmake_minimum_required(VERSION 3.16)

project(enzyme_consumer LANGUAGES C)

find_package(Enzyme REQUIRED CONFIG NO_DEFAULT_PATH)
message("found dir ${Enzyme_DIR}")
message("found ${Enzyme_FOUND}")
get_property(
  importTargetsAfter
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  PROPERTY IMPORTED_TARGETS)
message("imported targets ${importTargetsAfter}")

#add_subdirectory(${Enzyme_DIR} ${PROJECT_SOURCE_DIR}/build/enzyme)

add_executable(enzyme_test_1 main.c)
target_link_libraries(enzyme_test_1 PUBLIC ClangEnzymeFlags)

add_executable(enzyme_test_2 main.c)
target_compile_definitions(enzyme_test_2 PUBLIC -DNOVERSION)
target_link_libraries(enzyme_test_2 PUBLIC LLDEnzymeFlags)

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" "-DENZYME_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DTEST_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/relocated_llvm_test"
    "-DTEST_C_COMPILER=${CMAKE_C_COMPILER}" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/relocated_llvm_test.cmake"
  RESULT_VARIABLE relocated_llvm_result)
if(NOT relocated_llvm_result EQUAL 0)
  message(FATAL_ERROR "Relocated Enzyme_LLVM_BINARY_DIR test failed")
endif()
