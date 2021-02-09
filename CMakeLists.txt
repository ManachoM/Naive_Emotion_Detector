cmake_minimum_required(VERSION 3.17)
project(Naive_Emotion_Detector)

set(CMAKE_CXX_STANDARD 14)

include_directories(${OpenCV_INCLUDE_DIRS})



find_package(OpenCV REQUIRED)



add_executable(Naive_Emotion_Detector main.cpp)

target_link_libraries(Naive_Emotion_Detector ${OpenCV_LIBS})