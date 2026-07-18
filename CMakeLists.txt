cmake_minimum_required(VERSION 3.16)
project(fsmn_vad_kws LANGUAGES C CXX)

# ============================================================================
# FSMN-KWS + Silero VAD 纯 C++ 工程
#
# 与原 Makefile 等价：本地闭环打包 ONNXRuntime / PortAudio，产物为 mic_demo。
#   cmake -S . -B build && cmake --build build
#   ./build/mic_demo --verbose
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 打包的动态库是 x86_64；Apple Silicon 通过 Rosetta 运行，这里强制同一架构。
if(APPLE AND EXISTS "${CMAKE_SOURCE_DIR}/lib/libonnxruntime.1.dylib")
  execute_process(
    COMMAND file "${CMAKE_SOURCE_DIR}/lib/libonnxruntime.1.dylib"
    OUTPUT_VARIABLE ORT_FILE_INFO)
  if(ORT_FILE_INFO MATCHES "x86_64")
    set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "" FORCE)
  endif()
endif()

set(KNF_ROOT "${CMAKE_SOURCE_DIR}/third_party/kaldi-native-fbank")
set(KNF_CSRC "${KNF_ROOT}/kaldi-native-fbank/csrc")

set(KNF_SOURCES
  ${KNF_CSRC}/rfft.cc
  ${KNF_CSRC}/online-feature.cc
  ${KNF_CSRC}/feature-fbank.cc
  ${KNF_CSRC}/feature-window.cc
  ${KNF_CSRC}/log.cc
  ${KNF_CSRC}/feature-functions.cc
  ${KNF_CSRC}/mel-computations.cc
  ${KNF_CSRC}/fftsg.c)

# 工程源码（按阅读顺序）
set(APP_SOURCES
  main.cc
  common.cc
  mic_input.cc
  vad_detector.cc
  kws_engine.cc
  keyword_action.cc)

add_executable(mic_demo ${APP_SOURCES} ${KNF_SOURCES})

# 产物放回源码根目录，使 ./mic_demo 与相邻的 lib/、onnx_model/ 就地可用，
# 也让 @executable_path/lib 的 rpath 生效（与原 Makefile 一致）。
set_target_properties(mic_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_include_directories(mic_demo PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}
  ${KNF_ROOT})

# 只对工程源码开警告，第三方 kaldi-native-fbank 保持安静。
set_source_files_properties(${APP_SOURCES} PROPERTIES
  COMPILE_OPTIONS "-Wall;-Wextra")

target_link_libraries(mic_demo PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/libonnxruntime.1.dylib
  ${CMAKE_SOURCE_DIR}/lib/libportaudio.2.dylib)

if(APPLE)
  target_link_libraries(mic_demo PRIVATE
    "-framework CoreAudio"
    "-framework AudioToolbox"
    "-framework AudioUnit"
    "-framework CoreFoundation"
    "-framework CoreServices")
endif()

# 运行时从可执行文件同级 lib/ 查找依赖，与 Makefile 的 rpath 一致。
set_target_properties(mic_demo PROPERTIES
  BUILD_RPATH "@executable_path/lib"
  INSTALL_RPATH "@executable_path/lib")

# make run / make test 的等价目标。
add_custom_target(run
  COMMAND mic_demo --verbose
  DEPENDS mic_demo
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  USES_TERMINAL)

add_custom_target(run_test
  COMMAND mic_demo --keyword 大圣 --wav test/dasheng_1.wav --verbose
  DEPENDS mic_demo
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  USES_TERMINAL)
