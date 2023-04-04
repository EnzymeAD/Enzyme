cmake_minimum_required(VERSION 3.16)

project(enzyme_consumer LANGUAGES CXX)

find_package(Enzyme REQUIRED CONFIG)
#add_subdirectory(${Enzyme_DIR} ${PROJECT_SOURCE_DIR}/build/enzyme)

add_executable(enzyme_test_1 main.cpp)
target_link_libraries(enzyme_test_1 PUBLIC ClangEnzyme)

add_executable(enzyme_test_2 main.cpp)
target_link_libraries(enzyme_test_2 PUBLIC LLDEnzyme)
