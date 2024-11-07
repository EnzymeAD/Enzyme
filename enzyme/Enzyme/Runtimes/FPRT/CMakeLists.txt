
add_library( Enzyme-FPRT-GC-${LLVM_VERSION_MAJOR}
  GarbageCollection.cpp
)

target_include_directories(Enzyme-FPRT-GC-${LLVM_VERSION_MAJOR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
