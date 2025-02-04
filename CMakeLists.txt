cmake_minimum_required(VERSION 3.10)
project(SpinnakerExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED 10)
include_directories(include /opt/spinnaker/include/ ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(SPINNAKER_LIBS /opt/spinnaker/lib/libSpinnaker.so)
# Add executable
add_executable(spinnaker_example src/main.cpp src/FLIRCamera.cpp src/GPU.cpp)

# Link Spinnaker library
target_link_libraries(spinnaker_example ${SPINNAKER_LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Include Spinnaker headers
#target_include_directories(spinnaker_example PRIVATE ${Spinnaker_INCLUDE_DIRS})
