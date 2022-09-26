cmake_minimum_required(VERSION 3.14)
project(gradient-boosting-tree)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)

# Set compilers flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -std=c++17 --coverage -g -fno-exceptions -fno-inline -fsanitize=address")


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
  dataset_test
  tests/dataset_test.cpp
)
target_link_libraries(
  dataset_test
  GTest::gtest_main
)

add_executable(
  loss_test
  tests/loss_test.cpp
)
target_link_libraries(
  loss_test
  GTest::gtest_main
)

add_executable(
  model_test
  tests/model_test.cpp
)
target_link_libraries(
  model_test
  GTest::gtest_main
)

add_executable(
  tree_test
  tests/tree_test.cpp
)
target_link_libraries(
  tree_test
  GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(dataset_test)
gtest_discover_tests(loss_test)
gtest_discover_tests(tree_test)
gtest_discover_tests(model_test)


set(CMAKE_CXX_CLANG_TIDY 
  clang-tidy;
  -header-filter=.;
  -checks=*;
  -warnings-as-errors=*;
)