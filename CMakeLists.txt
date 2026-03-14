cmake_minimum_required(VERSION 3.20)
project(realtimestt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
if(APPLE)
  set(WHISPER_COREML ON CACHE BOOL "" FORCE)
  set(WHISPER_COREML_ALLOW_FALLBACK ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)

pybind11_add_module(_whisper_cpp_native native/whisper_cpp/pybind_module.cpp)
target_link_libraries(_whisper_cpp_native PRIVATE whisper)
target_include_directories(_whisper_cpp_native PRIVATE third_party/whisper.cpp/include)

install(TARGETS _whisper_cpp_native LIBRARY DESTINATION RealtimeSTT)
