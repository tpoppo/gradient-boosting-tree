cmake_minimum_required(VERSION 3.14)
project(gradient-boosting-tree)

option(RELEASE "Disable debugging options" OFF)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)


# Set compilers flags
set(CMAKE_CXX_FLAGS "-std=c++17")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wfloat-equal -Wformat=2")

if(RELEASE)
  message("Release Mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -Ofast -g -faggressive-loop-optimizations -march=native")
else()
  message("Debugging Mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -g -fno-exceptions -fno-inline -fsanitize=address")
endif()

# set(CMAKE_CXX_CLANG_TIDY
#   clang-tidy;
#   -format-style='file';
#   -header-filter=${CMAKE_CURRENT_SOURCE_DIR};
# )


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