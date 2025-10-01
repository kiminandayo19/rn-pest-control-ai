cmake_minimum_required(VERSION 3.13)

# Define the library name here.
project(appmodules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-this-capture")

# Root dirs
set(ROOT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv/sdk/native/jni)
set(ONNXRUNTIME_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include)
set(RNVC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../node_modules/react-native-vision-camera/android/src/main/cpp)

# Find packages
find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)
find_package(react-native-vision-camera REQUIRED CONFIG)
find_package(OpenCV REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${REACT_NATIVE_DIR}/ReactAndroid/cmake-utils")
include(${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
  ../../../../../shared/NativeImageProcessor.cpp
  ${ROOT_DIRS}/node_modules/react-native-vision-camera/android/src/main/cpp/frameprocessors/FrameHostObject.cpp
  ${ROOT_DIRS}/node_modules/react-native-vision-camera/android/src/main/cpp/frameprocessors/java-bindings/JFrame.cpp
)
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ONNXRUNTIME_ROOT_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so
)

# Lookup headers
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
  ${ROOT_DIRS}/shared
  ${ONNXRUNTIME_ROOT_DIR}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/opencv/sdk/native/jni/include
  ${ROOT_DIRS}/node_modules/react-native/React
  ${ROOT_DIRS}/node_modules/react-native/React/Base
  ${ROOT_DIRS}/node_modules/react-native/ReactCommon/jsi
  ${RNVC_DIR}
  ${ROOT_DIRS}/node_modules/react-native-vision-camera/android/src/main/cpp/frameprocessors
  ${ROOT_DIRS}/node_modules/react-native-vision-camera/android/src/main/cpp/frameprocessors/java-bindings
)

target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  ${OpenCV_LIBS}
  onnxruntime
  android
  ReactAndroid::jsi
  fbjni::fbjni
  react-native-vision-camera::VisionCamera
)
