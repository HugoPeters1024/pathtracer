cmake_minimum_required(VERSION 3.16)
project(pathtracer)

set(CMAKE_CXX_STANDARD 14)

add_executable(pathtracer main.cpp)

find_package(glfw3 3.3.2 REQUIRED)
target_link_libraries(pathtracer glfw)
find_package(OpenGL REQUIRED)
# These variables have been set by the previous command
target_include_directories(pathtracer PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(pathtracer ${OPENGL_LIBRARIES})
