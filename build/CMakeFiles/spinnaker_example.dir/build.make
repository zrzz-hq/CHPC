# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/Desktop/CHPC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Desktop/CHPC/build

# Include any dependencies generated for this target.
include CMakeFiles/spinnaker_example.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/spinnaker_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/spinnaker_example.dir/flags.make

CMakeFiles/spinnaker_example.dir/src/main.cpp.o: CMakeFiles/spinnaker_example.dir/flags.make
CMakeFiles/spinnaker_example.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/CHPC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/spinnaker_example.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/spinnaker_example.dir/src/main.cpp.o -c /home/nvidia/Desktop/CHPC/src/main.cpp

CMakeFiles/spinnaker_example.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spinnaker_example.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/CHPC/src/main.cpp > CMakeFiles/spinnaker_example.dir/src/main.cpp.i

CMakeFiles/spinnaker_example.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spinnaker_example.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/CHPC/src/main.cpp -o CMakeFiles/spinnaker_example.dir/src/main.cpp.s

CMakeFiles/spinnaker_example.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/spinnaker_example.dir/src/main.cpp.o.requires

CMakeFiles/spinnaker_example.dir/src/main.cpp.o.provides: CMakeFiles/spinnaker_example.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/spinnaker_example.dir/build.make CMakeFiles/spinnaker_example.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/spinnaker_example.dir/src/main.cpp.o.provides

CMakeFiles/spinnaker_example.dir/src/main.cpp.o.provides.build: CMakeFiles/spinnaker_example.dir/src/main.cpp.o


CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o: CMakeFiles/spinnaker_example.dir/flags.make
CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o: ../src/FLIRCamera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/CHPC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o -c /home/nvidia/Desktop/CHPC/src/FLIRCamera.cpp

CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/CHPC/src/FLIRCamera.cpp > CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.i

CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/CHPC/src/FLIRCamera.cpp -o CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.s

CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o.requires:

.PHONY : CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o.requires

CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o.provides: CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o.requires
	$(MAKE) -f CMakeFiles/spinnaker_example.dir/build.make CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o.provides.build
.PHONY : CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o.provides

CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o.provides.build: CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o


CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o: CMakeFiles/spinnaker_example.dir/flags.make
CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o: ../src/GPU.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/CHPC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o -c /home/nvidia/Desktop/CHPC/src/GPU.cpp

CMakeFiles/spinnaker_example.dir/src/GPU.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spinnaker_example.dir/src/GPU.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/CHPC/src/GPU.cpp > CMakeFiles/spinnaker_example.dir/src/GPU.cpp.i

CMakeFiles/spinnaker_example.dir/src/GPU.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spinnaker_example.dir/src/GPU.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/CHPC/src/GPU.cpp -o CMakeFiles/spinnaker_example.dir/src/GPU.cpp.s

CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o.requires:

.PHONY : CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o.requires

CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o.provides: CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o.requires
	$(MAKE) -f CMakeFiles/spinnaker_example.dir/build.make CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o.provides.build
.PHONY : CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o.provides

CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o.provides.build: CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o


# Object files for target spinnaker_example
spinnaker_example_OBJECTS = \
"CMakeFiles/spinnaker_example.dir/src/main.cpp.o" \
"CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o" \
"CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o"

# External object files for target spinnaker_example
spinnaker_example_EXTERNAL_OBJECTS =

spinnaker_example: CMakeFiles/spinnaker_example.dir/src/main.cpp.o
spinnaker_example: CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o
spinnaker_example: CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o
spinnaker_example: CMakeFiles/spinnaker_example.dir/build.make
spinnaker_example: /opt/spinnaker/lib/libSpinnaker.so
spinnaker_example: /usr/local/cuda-10.0/lib64/libcudart_static.a
spinnaker_example: /usr/lib/aarch64-linux-gnu/librt.so
spinnaker_example: CMakeFiles/spinnaker_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Desktop/CHPC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable spinnaker_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spinnaker_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/spinnaker_example.dir/build: spinnaker_example

.PHONY : CMakeFiles/spinnaker_example.dir/build

CMakeFiles/spinnaker_example.dir/requires: CMakeFiles/spinnaker_example.dir/src/main.cpp.o.requires
CMakeFiles/spinnaker_example.dir/requires: CMakeFiles/spinnaker_example.dir/src/FLIRCamera.cpp.o.requires
CMakeFiles/spinnaker_example.dir/requires: CMakeFiles/spinnaker_example.dir/src/GPU.cpp.o.requires

.PHONY : CMakeFiles/spinnaker_example.dir/requires

CMakeFiles/spinnaker_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/spinnaker_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/spinnaker_example.dir/clean

CMakeFiles/spinnaker_example.dir/depend:
	cd /home/nvidia/Desktop/CHPC/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Desktop/CHPC /home/nvidia/Desktop/CHPC /home/nvidia/Desktop/CHPC/build /home/nvidia/Desktop/CHPC/build /home/nvidia/Desktop/CHPC/build/CMakeFiles/spinnaker_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/spinnaker_example.dir/depend

