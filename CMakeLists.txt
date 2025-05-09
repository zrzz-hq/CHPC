cmake_minimum_required(VERSION 3.10)
project(SpinnakerExample LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED 10)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(
    /opt/spinnaker/include/ 
    include 
    ${CUDA_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS} 
    ${OpenGL_INCLUDE_DIRS}
    imgui
)

add_subdirectory(ImGuiFileDialog)

set(SPINNAKER_LIBS /opt/spinnaker/lib/libSpinnaker.so)

set_source_files_properties(src/GPU.cpp PROPERTIES LANGUAGE CUDA)

#ImGui Library
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
add_executable(PhaseVisualizer
    src/cnpy.cpp
    src/main.cpp 
    src/FLIRCamera.cpp 
    src/GPU.cpp 
    src/kernel.cu 
    src/ui.cpp
)

# Link Spinnaker library
target_link_libraries(PhaseVisualizer 
    ${SPINNAKER_LIBS} 
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBS}
    ImGuiFileDialog
    Boost::filesystem
    Boost::system
    OpenGL::GL 
    glfw
    imgui
    ZLIB::ZLIB
)
