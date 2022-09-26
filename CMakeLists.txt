cmake_minimum_required(VERSION 3.14)
project(gradient-boosting-tree)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)

# Set compilers flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wlogical-op -Wshift-overflow=2 -std=c++17 --coverage -g -fno-exceptions -fno-inline -fsanitize=address")


include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  tests
  tests/tests.cpp
)
target_link_libraries(
  tests
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)



set(CMAKE_CXX_CLANG_TIDY 
  clang-tidy;
  -header-filter=.;
  -checks=*;
  -warnings-as-errors=*;
)