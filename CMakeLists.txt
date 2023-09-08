cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(lala-pc
  HOMEPAGE_URL "https://github.com/lattice-land/lala-pc"
  LANGUAGES CUDA CXX)
option(GPU "GPU" ON)
option(LOCAL_DEPS "LOCAL_DEPS" ON)
option(LALA_PC_BUILD_TESTS "LALA_PC_BUILD_TESTS" OFF)

# Dependencies

include(FetchContent)

if(LOCAL_DEPS)
  FetchContent_Declare(lala_core SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-core/")
else()
  FetchContent_Declare(
    lala_core
    GIT_REPOSITORY "https://github.com/lattice-land/lala-core.git"
    GIT_TAG        v1.0.0
  )
endif()

FetchContent_MakeAvailable(lala_core)

# Preparing the library

add_library(lala_pc INTERFACE)
target_link_libraries(lala_pc INTERFACE lala_core)
target_include_directories(lala_pc INTERFACE include)

if(LALA_PC_BUILD_TESTS)

# Google Test dependency

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

if(LOCAL_DEPS)
  FetchContent_Declare(lala_parsing SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-parsing/")
else()
  FetchContent_Declare(
    lala_parsing
    GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
    GIT_TAG        v1.0.0
  )
endif()
FetchContent_MakeAvailable(lala_parsing)

add_executable(pc_test tests/pc_test.cpp)
target_include_directories(pc_test PRIVATE "${lala-core_SOURCE_DIR}/tests/include")
target_link_libraries(pc_test lala_parsing lala_pc gtest_main)
gtest_discover_tests(pc_test)

endif()

# Documentation
find_package(Doxygen REQUIRED doxygen)

set(DOXYGEN_PROJECT_NAME "Lattice land propagators completion library")
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CASE_SENSE_NAMES NO)
set(DOXYGEN_CLASS_DIAGRAMS NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE NO)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_SORT_BY_SCOPE_NAME YES)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_STRIP_CODE_COMMENTS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRA_PACKAGES stmaryrd)
set(DOXYGEN_EXCLUDE_SYMBOLS = impl)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_PREDEFINED __NVCC__)
set(DOXYGEN_REFERENCED_BY_RELATION NO)
set(DOXYGEN_REFERENCES_RELATION NO)

doxygen_add_docs(doc_lala_pc
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")
