cmake_minimum_required(VERSION 3.16)

project(CppInterfaceTests LANGUAGES CXX)

file(GLOB_RECURSE cpp_tests ${PROJECT_SOURCE_DIR}/*.cpp)

enable_testing()

foreach(filename ${cpp_tests})		
  get_filename_component(testname ${filename} NAME_WE)
  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC ClangEnzymeFlags)
  add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		
endforeach(filename ${cpp_tests})