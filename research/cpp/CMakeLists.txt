cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nationgrid)
set(Python_ADDITIONAL_VERSIONS 3.6)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonLibs 3 REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

if(NOT PYTHONLIBS_FOUND)
    message(FATAL_ERROR "Python3 Not Found!")
endif(NOT PYTHONLIBS_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


message(STATUS "Python3 status:")
message(STATUS "    version:${PYTHONLIBS_VERSION_STRING}")
message(STATUS "    libraries:${PYTHON_LIBRARY}")
message(STATUS "    include path:${PYTHON_INCLUDE_DIR}")

add_executable(nationgrid libtorch_classifier_v1.cpp)
target_link_libraries(nationgrid ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(nationgrid PRIVATE ${PYTHON_INCLUDE_DIR})
set_property(TARGET nationgrid PROPERTY CXX_STANDARD 14)