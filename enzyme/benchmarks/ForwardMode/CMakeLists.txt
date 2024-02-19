# Run regression and unit tests
add_lit_testsuite(bench-enzyme-forward "Running enzyme forward mode benchmarks tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ENZYME_BENCH_DEPS}
    ARGS -v -j 1
)

set_target_properties(bench-enzyme-forward PROPERTIES FOLDER "bench Tests")

# add_subdirectory(taylorlog)
# add_subdirectory(ode)
# add_subdirectory(ode-const)

add_subdirectory(gmm)
add_subdirectory(ba)
add_subdirectory(hand)
add_subdirectory(lstm)
