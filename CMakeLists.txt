cmake_minimum_required(VERSION 2.7)

project(openmit)
find_package(OpenMP)

message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -g -O3 -fPIC -Wall -mssse3")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Make sure compiler-specific support C++11
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
	if (NOT (GCC_VERSION VERSION_GREATER 4.6 OR GCC_VERSION VERSION_EQUAL 4.6))
		message(FATAL_ERROR "${PROJECT_NAME} project requires g++ 4.6 or greater.")
  else ()
    message(STATUS "-std=c++ ...")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
	endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
else ()
	message(FATAL_ERROR "Your C++ compiler does not support C++11.")
endif ()

# Dependenced include && library
#${PROJECT_SOURCE_DIR}/third_party/dmlc-core/include
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/include
  ${PROJECT_SOURCE_DIR}/third_party/hadoop/include
  ${PROJECT_SOURCE_DIR}/third_party/ps-lite/deps/include 
  ${PROJECT_SOURCE_DIR}/third_party/makefile/gen/grpc/include 
  ${PROJECT_SOURCE_DIR}/tools/grpc/helloworld
  $ENV{JAVA_HOME}/include)

link_directories(
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/third_party/lib
  ${PROJECT_SOURCE_DIR}/third_party/hadoop/lib
  ${PROJECT_SOURCE_DIR}/third_party/ps-lite/deps/lib
  ${PROJECT_SOURCE_DIR}/third_party/makefile/gen/grpc/lib
  $ENV{JAVA_HOME}/jre/lib/amd64/server)
  #${PROJECT_SOURCE_DIR}/third_party/makefile/grpc/libs/opt

set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

# Add Subdirectory
#add_subdirectory(openmit)
#add_subdirectory(test)
add_subdirectory(tools/grpc)
