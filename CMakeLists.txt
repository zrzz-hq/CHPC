cmake_minimum_required(VERSION 3.10)
project(SpinnakerExample LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED 10)
find_package(Boost REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS OpenGL Widgets Gui)


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

include_directories(include /opt/spinnaker/include/ ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${X11_INCLUDE_DIR} ${OpenGL_INCLUDE_DIRS})


set(SPINNAKER_LIBS /opt/spinnaker/lib/libSpinnaker.so)

set_source_files_properties(src/GPU.cpp PROPERTIES LANGUAGE CUDA)

add_library(gpu src/GPU.cpp src/kernel.cu)
target_link_libraries(gpu ${CUDA_LIBRARIES})

# Add executable
add_executable(spinnaker_example src/main.cpp src/FLIRCamera.cpp)

# Link Spinnaker library
target_link_libraries(spinnaker_example 
    Qt5::OpenGL
    Qt5::Gui 
    Qt5::Widgets
    ${SPINNAKER_LIBS} 
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBS}
    gpu 
    OpenGL::GL 
)

# Include Spinnaker headers
#target_include_directories(spinnaker_example PRIVATE ${Spinnaker_INCLUDE_DIRS})
