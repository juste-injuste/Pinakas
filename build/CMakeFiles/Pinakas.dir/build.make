# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.26

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\build"

# Include any dependencies generated for this target.
include CMakeFiles/Pinakas.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Pinakas.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Pinakas.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Pinakas.dir/flags.make

CMakeFiles/Pinakas.dir/src/Pinakas.cpp.obj: CMakeFiles/Pinakas.dir/flags.make
CMakeFiles/Pinakas.dir/src/Pinakas.cpp.obj: CMakeFiles/Pinakas.dir/includes_CXX.rsp
CMakeFiles/Pinakas.dir/src/Pinakas.cpp.obj: C:/Users/justi/OneDrive\ -\ USherbrooke/Documents/github/Pinakas/src/Pinakas.cpp
CMakeFiles/Pinakas.dir/src/Pinakas.cpp.obj: CMakeFiles/Pinakas.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Pinakas.dir/src/Pinakas.cpp.obj"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Pinakas.dir/src/Pinakas.cpp.obj -MF CMakeFiles\Pinakas.dir\src\Pinakas.cpp.obj.d -o CMakeFiles\Pinakas.dir\src\Pinakas.cpp.obj -c "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\src\Pinakas.cpp"

CMakeFiles/Pinakas.dir/src/Pinakas.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Pinakas.dir/src/Pinakas.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\src\Pinakas.cpp" > CMakeFiles\Pinakas.dir\src\Pinakas.cpp.i

CMakeFiles/Pinakas.dir/src/Pinakas.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Pinakas.dir/src/Pinakas.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\src\Pinakas.cpp" -o CMakeFiles\Pinakas.dir\src\Pinakas.cpp.s

# Object files for target Pinakas
Pinakas_OBJECTS = \
"CMakeFiles/Pinakas.dir/src/Pinakas.cpp.obj"

# External object files for target Pinakas
Pinakas_EXTERNAL_OBJECTS =

libPinakas.a: CMakeFiles/Pinakas.dir/src/Pinakas.cpp.obj
libPinakas.a: CMakeFiles/Pinakas.dir/build.make
libPinakas.a: CMakeFiles/Pinakas.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libPinakas.a"
	$(CMAKE_COMMAND) -P CMakeFiles\Pinakas.dir\cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Pinakas.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Pinakas.dir/build: libPinakas.a
.PHONY : CMakeFiles/Pinakas.dir/build

CMakeFiles/Pinakas.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Pinakas.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Pinakas.dir/clean

CMakeFiles/Pinakas.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas" "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas" "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\build" "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\build" "C:\Users\justi\OneDrive - USherbrooke\Documents\github\Pinakas\build\CMakeFiles\Pinakas.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Pinakas.dir/depend

