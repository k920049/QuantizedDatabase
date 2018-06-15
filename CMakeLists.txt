cmake_minimum_required(VERSION 3.10)
project(Quantization_Database)
# We'll use C++11
set(CMAKE_CXX_STANDARD 11)
set(PYTHON_LIBRARY /usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/libpython3.6.dylib)
set(PYTHON_INCLUDE_DIR /usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/include/python3.6m)
set(PYTHON_EXECUTABLE /usr/local/Cellar/python/3.6.5/bin/python3)
set(USE_PYTHON2 OFF)
# To load some library not on the list
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake/common)
# Here are some libraries that I've used for the project
find_package(OPENCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonInterp 3.6 EXACT REQUIRED)
find_package(Numpy REQUIRED)
# Header files
include_directories(./header)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

add_executable(Quantization_Database main.cpp header/data_read.h)
# header/database.h src/database/database.cpp header/knn.h src/database/knn.cpp header/data_read.h)
target_link_libraries(Quantization_Database ${OpenCV_LIBS})
target_link_libraries(Quantization_Database ${PYTHON_LIBRARIES})

message(STATUS "dummy")

message(STATUS ${PYTHON_NUMPY_INCLUDE_DIR})
message(STATUS ${PYTHON_INCLUDE_DIRS})