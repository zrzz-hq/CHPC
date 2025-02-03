cmake_minimum_required(VERSION 3.10)
project(SpinnakerExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED 10)
include_directories(/opt/spinnaker/include/ ${CUDA_INCLUDE_DIRS})
# Add executable
add_executable(spinnaker_example main.cpp FLIRCamera.cpp GPU.cpp)

# Link Spinnaker library
target_link_libraries(spinnaker_example /opt/spinnaker/lib/libSpinnaker.so
					/opt/spinnaker/lib/libSpinVideo.so
					${CUDA_LIBRARIES})

# Include Spinnaker headers
#target_include_directories(spinnaker_example PRIVATE ${Spinnaker_INCLUDE_DIRS})
