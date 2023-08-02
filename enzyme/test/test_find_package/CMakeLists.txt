cmake_minimum_required(VERSION 3.16)

project(enzyme_consumer LANGUAGES C)

find_package(Enzyme REQUIRED CONFIG NO_DEFAULT_PATH)
message("found dir ${Enzyme_DIR}")
message("found ${Enzyme_FOUND}")
get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
message("imported targets ${importTargetsAfter}")

#add_subdirectory(${Enzyme_DIR} ${PROJECT_SOURCE_DIR}/build/enzyme)

add_executable(enzyme_test_1 main.c)
target_link_libraries(enzyme_test_1 PUBLIC ClangEnzymeFlags)

if (${Enzyme_LLVM_VERSION_MAJOR} GREATER 10)
    add_executable(enzyme_test_2 main.c)
    target_compile_definitions(enzyme_test_2 PUBLIC -DNOVERSION)
    target_link_libraries(enzyme_test_2 PUBLIC LLDEnzymeFlags)
endif()
