cmake_minimum_required(VERSION 3.0)

project(Pinakas)

# Set the source files for the library
set(LIB_SOURCE_FILES src/Pinakas.cpp)

# Set the header files for the library
set(LIB_HEADER_FILES include/Pinakas.hpp)

# Create the library target
add_library(Pinakas ${LIB_SOURCE_FILES} ${LIB_HEADER_FILES})

# Set the include directories for the library
target_include_directories(Pinakas PUBLIC include)

# Set the source files for the executable
set(EXE_SOURCE_FILES test/main.cpp)

# Create the executable target
add_executable(Pinakas_main ${EXE_SOURCE_FILES})

# Link the executable to the library
target_link_libraries(Pinakas_main Pinakas)