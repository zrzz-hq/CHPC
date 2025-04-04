cmake_minimum_required(VERSION 3.10)
project(PhaseVisualizer LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED 10)
find_package(Boost REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

include_directories(
    include
    imgui
    /opt/spinnaker/include/ 
    ${CUDA_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS} 
    ${OpenGL_INCLUDE_DIRS}
)


set(SPINNAKER_LIBS /opt/spinnaker/lib/libSpinnaker.so)

set_source_files_properties(src/GPU.cpp PROPERTIES LANGUAGE CUDA)

add_library(gpu src/GPU.cpp src/kernel.cu)
target_link_libraries(gpu ${CUDA_LIBRARIES})

add_library(imgui
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
)

target_link_libraries(imgui OpenGL::GL glfw)

# Add executable
add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/FLIRCamera.cpp
    src/ui.cpp
)

# Link Spinnaker library
target_link_libraries(${PROJECT_NAME} 
    ${SPINNAKER_LIBS} 
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBS}
    glfw
    imgui
    gpu 
    OpenGL::GL 
)

# Include Spinnaker headers
#target_include_directories(spinnaker_example PRIVATE ${Spinnaker_INCLUDE_DIRS})
