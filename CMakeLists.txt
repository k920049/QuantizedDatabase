cmake_minimum_required(VERSION 3.10)
project(Quantization_Database)
# We'll use C++11
set(CMAKE_CXX_STANDARD 11)
# Here are some libraries that I've used for the project
find_package(OPENCV REQUIRED)
# Header files
include_directories(./header)

add_executable(Quantization_Database main.cpp header/database.h src/database/database.cpp header/knn.h src/database/knn.cpp)
target_link_libraries(Quantization_Database ${OpenCV_LIBS})