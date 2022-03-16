cmake_minimum_required(VERSION 3.20)

include(FetchContent)
include(CMakePrintHelpers)

project(lala-ipc LANGUAGES CUDA CXX)
option(GPU "GPU" ON)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# A new configuration mode "LDebug" for light debug.

if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND CMAKE_CONFIGURATION_TYPES LDebug)
  list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
    "Add a light debug configuration (LDebug)"
    FORCE)
endif()

# Cuda-battery dependency

FetchContent_Declare(
  cuda_battery
  GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
  GIT_TAG        337cf303a190dfd500cfc7e960a94fe9cfcc541c
)
FetchContent_MakeAvailable(cuda_battery)

FetchContent_Declare(
  lala_core
  GIT_REPOSITORY "https://github.com/lattice-land/lala-core.git"
  GIT_TAG        2e4fa45ddb32aa31dea423cf84130bf43f0dab74
)
FetchContent_MakeAvailable(lala_core)

# Google Test dependency

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Thrust dependency

FetchContent_Declare(
  thrust
  GIT_REPOSITORY "https://github.com/NVIDIA/thrust.git"
  GIT_TAG        1.12.0
)
FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
    FetchContent_Populate(thrust)
endif()

# Preparing the library

add_library(lala_ipc INTERFACE)
target_include_directories(lala_ipc INTERFACE include)

if(GPU)
    set_property(TARGET lala_ipc PROPERTY CUDA_ARCHITECTURES 75)
    set_property(TARGET lala_ipc APPEND PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_compile_options(lala_ipc INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -rdc=true>
    $<$<CONFIG:Debug>:-g -DDEBUG>
    $<$<CONFIG:LDebug>:-g -DTRACE>
)

enable_testing()
include(GoogleTest)

add_executable(ipc_test tests/ipc_test.cpp)
target_link_libraries(ipc_test cuda_battery lala_core lala_ipc gtest_main)
target_include_directories(ipc_test PRIVATE ${cuda_battery_SOURCE_DIR}/include ${thrust_SOURCE_DIR})
gtest_discover_tests(ipc_test)

# Documentation

find_package(Doxygen REQUIRED doxygen)

# Options based on those of Halide: https://github.com/halide/Halide/blob/feb81a2a0f7bc39cfa499e4e1a7dfedbc5250aaa/doc/CMakeLists.txt
set(DOXYGEN_PROJECT_NAME "Interval propagators completion (IPC)")
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CASE_SENSE_NAMES NO)
set(DOXYGEN_CLASS_DIAGRAMS NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_REFERENCED_BY_RELATION YES)
set(DOXYGEN_REFERENCES_RELATION YES)
set(DOXYGEN_SORT_BY_SCOPE_NAME YES)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_STRIP_CODE_COMMENTS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRA_PACKAGES stmaryrd)

doxygen_add_docs(doc_lalaipc
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")
