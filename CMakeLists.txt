cmake_minimum_required(VERSION 3.10)
project(SpinnakerExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Spinnaker SDK
find_package(Spinnaker REQUIRED)

# Add executable
add_executable(spinnaker_example main.cpp)

# Link Spinnaker library
target_link_libraries(spinnaker_example PRIVATE Spinnaker::Spinnaker)

# Include Spinnaker headers
target_include_directories(spinnaker_example PRIVATE ${Spinnaker_INCLUDE_DIRS})
