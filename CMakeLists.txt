cmake_minimum_required(VERSION 3.15)

project(robot_trailer_cpp_modules LANGUAGES CXX)


set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)

# Find Python and pybind11.  pybind11 is listed as a Pixi dependency, so it will
# be available in the environment.
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the pybind11 module from the C++ source.
pybind11_add_module(robot_trailer_sim_cpp
    cpp/simulator.cpp
)

# Build a combined module that provides both the simulator and the planner.
pybind11_add_module(robot_trailer_cpp
    cpp/simulator.cpp
    cpp/planner.cpp
)

target_compile_options(robot_trailer_cpp PRIVATE
    -O2
    -g
)

target_compile_options(robot_trailer_sim_cpp PRIVATE
    -O2
    -g
)

# Ensure the module is placed next to the Python package when built.
set_target_properties(robot_trailer_sim_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(robot_trailer_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

