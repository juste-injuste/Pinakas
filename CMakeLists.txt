cmake_minimum_required(VERSION 3.0)

project(Pinakas)

# Set the source files
set(SOURCE_FILES test/main.cpp src/Pinakas.cpp)

# Set the include directories
include_directories(include)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})