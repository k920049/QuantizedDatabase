cmake_minimum_required(VERSION 3.10)
project(Quantization_Database)
# We'll use C++11
set(CMAKE_CXX_STANDARD 11)
# To load some library not on the list
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake/common)
# Here are some libraries that I've used for the project
find_package(OPENCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonInterp 2.7.15 EXACT REQUIRED)
# Header files
include_directories(./header)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(Quantization_Database main.cpp header/data_read.h header/feature_extract.h)
# header/database.h src/database/database.cpp header/knn.h src/database/knn.cpp header/data_read.h)
target_link_libraries(Quantization_Database ${OpenCV_LIBS})
target_link_libraries(Quantization_Database ${PYTHON_LIBRARIES})
message(STATUS ${PYTHON_INCLUDE_DIRS})