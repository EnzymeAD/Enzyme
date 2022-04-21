set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${Clang_FOUND})
    message("starting ext project")
include(ExternalProject)
ExternalProject_Add(gsl
    GIT_REPOSITORY https://github.com/ampl/gsl
    GIT_TAG 6e40fb13a501393f3a9deb7c4fcbee85241a0339
    PREFIX gsl
    BUILD_IN_SOURCE 1
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/gsl/install
    CONFIGURE_COMMAND autoreconf -i && ./configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/gsl/install CC=$<TARGET_FILE:clang> 
    BUILD_COMMAND  sh -c "make -C gsl && $<TARGET_FILE:clang> cblas/*.c -I . -S -emit-llvm -O1"
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    TEST_COMMAND      ""
)
set_target_properties(gsl PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_custom_target(blasheaders cp "${CMAKE_CURRENT_SOURCE_DIR}/makeblas.cmake" "${CMAKE_CURRENT_BINARY_DIR}/gsl/CMakeLists.txt" && cd "${CMAKE_CURRENT_BINARY_DIR}/gsl" && ${CMAKE_COMMAND} . DEPENDS gsl ${CMAKE_CURRENT_SOURCE_DIR}/makeblas.cmake)
set_target_properties(blasheaders PROPERTIES EXCLUDE_FROM_ALL TRUE)

if (${LLVM_VERSION_MAJOR} LESS 8)
    add_llvm_loadable_module( BCPass-${LLVM_VERSION_MAJOR}
        BCLoader.cpp ClangBCLoader.cpp
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
else()
# on windows `PLUGIN_TOOL` doesn't link against LLVM.dll
if ((WIN32 OR CYGWIN) AND LLVM_LINK_LLVM_DYLIB)
    add_llvm_library( BCPass-${LLVM_VERSION_MAJOR}
        BCLoader.cpp ClangBCLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
	LINK_COMPONENTS
	LLVM
    )
else()
    add_llvm_library( BCPass-${LLVM_VERSION_MAJOR}
        BCLoader.cpp ClangBCLoader.cpp
        MODULE
        DEPENDS
        intrinsics_gen
        PLUGIN_TOOL
        opt
    )
endif()
endif()

add_dependencies(BCPass-${LLVM_VERSION_MAJOR} blasheaders)
target_include_directories(BCPass-${LLVM_VERSION_MAJOR} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/gsl)

if (APPLE)
# Darwin-specific linker flags for loadable modules.
set_target_properties(BCPass-${LLVM_VERSION_MAJOR} PROPERTIES
    LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()

set_target_properties(BCPass-${LLVM_VERSION_MAJOR} PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
