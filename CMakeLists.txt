cmake_minimum_required(VERSION 3.10)
project(SpinnakerExample LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED 10)
find_package(Boost REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(include /opt/spinnaker/include/ ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${X11_INCLUDE_DIR} ${OpenGL_INCLUDE_DIRS})

set(SPINNAKER_LIBS /opt/spinnaker/lib/libSpinnaker.so)

set_source_files_properties(src/GPU.cpp PROPERTIES LANGUAGE CUDA)
# Add executable
add_executable(spinnaker_example src/main.cpp src/FLIRCamera.cpp src/GPU.cpp src/kernel.cu)

# Link Spinnaker library
target_link_libraries(spinnaker_example ${SPINNAKER_LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${X11_LIBRARIES} OpenGL::GL)

# Include Spinnaker headers
#target_include_directories(spinnaker_example PRIVATE ${Spinnaker_INCLUDE_DIRS})
