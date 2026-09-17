cmake_minimum_required(VERSION 3.16)
project(opencv_pipeline_reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(opencv_pipeline opencv_pipeline.cpp)
target_link_libraries(opencv_pipeline PRIVATE ${OpenCV_LIBS})
