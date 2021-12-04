cmake_minimum_required(VERSION 3.16)
project(detect CXX)
set(CMAKE_CXX_STANDARD 14)

set(TENSORFLOW_SOURCE_DIR "/home/pavel/tensorflow-2.5.0")

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

find_package(OpenCV REQUIRED)

include_directories(detect
    "${TENSORFLOW_SOURCE_DIR}"
    "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include")

add_executable(detect detection.cxx)

target_link_libraries(detect
    ${OpenCV_LIBS}
    "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a"
    "/usr/lib/x86_64-linux-gnu/libedgetpu.so.1.0"
    -lpthread 
    -lm 
    -ldl
)