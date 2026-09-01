# Run regression and unit tests
add_lit_testsuite(check-typeanalysis "Running enzyme regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-typeanalysis PROPERTIES FOLDER "Tests")

if (ENZYME_BUILD_TYPETREE_BENCHMARK)
    find_package(benchmark CONFIG REQUIRED)
    add_executable(typetree-benchmark TypeTreeBenchmark.cpp)
    target_include_directories(typetree-benchmark PRIVATE ${ENZYME_SOURCE_DIR})
    target_link_libraries(typetree-benchmark PRIVATE benchmark::benchmark LLVMSupport)
    set_target_properties(typetree-benchmark PROPERTIES FOLDER "Tests")
endif()
