# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /home/hugo/Downloads/CLion-2019.1.3/clion-2019.1.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/hugo/Downloads/CLion-2019.1.3/clion-2019.1.3/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hugo/repos/pathtracer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hugo/repos/pathtracer

# Include any dependencies generated for this target.
include CMakeFiles/pathtracer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pathtracer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pathtracer.dir/flags.make

CMakeFiles/pathtracer.dir/main.cpp.o: CMakeFiles/pathtracer.dir/flags.make
CMakeFiles/pathtracer.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hugo/repos/pathtracer/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pathtracer.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pathtracer.dir/main.cpp.o -c /home/hugo/repos/pathtracer/main.cpp

CMakeFiles/pathtracer.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pathtracer.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hugo/repos/pathtracer/main.cpp > CMakeFiles/pathtracer.dir/main.cpp.i

CMakeFiles/pathtracer.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pathtracer.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hugo/repos/pathtracer/main.cpp -o CMakeFiles/pathtracer.dir/main.cpp.s

# Object files for target pathtracer
pathtracer_OBJECTS = \
"CMakeFiles/pathtracer.dir/main.cpp.o"

# External object files for target pathtracer
pathtracer_EXTERNAL_OBJECTS =

pathtracer: CMakeFiles/pathtracer.dir/main.cpp.o
pathtracer: CMakeFiles/pathtracer.dir/build.make
pathtracer: /usr/local/lib/libglfw3.a
pathtracer: /usr/lib/x86_64-linux-gnu/libGL.so
pathtracer: /usr/lib/x86_64-linux-gnu/libGLU.so
pathtracer: /usr/lib/x86_64-linux-gnu/librt.so
pathtracer: /usr/lib/x86_64-linux-gnu/libm.so
pathtracer: /usr/lib/x86_64-linux-gnu/libX11.so
pathtracer: CMakeFiles/pathtracer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hugo/repos/pathtracer/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pathtracer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pathtracer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pathtracer.dir/build: pathtracer

.PHONY : CMakeFiles/pathtracer.dir/build

CMakeFiles/pathtracer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pathtracer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pathtracer.dir/clean

CMakeFiles/pathtracer.dir/depend:
	cd /home/hugo/repos/pathtracer && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hugo/repos/pathtracer /home/hugo/repos/pathtracer /home/hugo/repos/pathtracer /home/hugo/repos/pathtracer /home/hugo/repos/pathtracer/CMakeFiles/pathtracer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pathtracer.dir/depend

