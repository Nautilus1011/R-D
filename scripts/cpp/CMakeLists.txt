cmake_minimum_required(VERSION 3.10)
project(RealtimeObjectDetection)

set(CMAKE_CXX_STANDARD 17)

# OpenCVを探す
# OpenCVはfind_packageに対応しているからこれができるみたい
find_package(OpenCV REQUIRED)

# ONNX Runtimeの場所（環境に合わせてパスを書き換えてください）
# ONNX Runtimeはまだ対応していないらしいからヘッダーファイルのパスを指定
set(ONNXRUNTIME_DIR "/path/to/onnxruntime-linux-x64-1.18.0")  # 例

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

# ソースファイルを指定（例としてmain.cppのみ）
add_executable(detect main.cpp)

# ライブラリをリンク
target_link_libraries(detect
    ${OpenCV_LIBS}
    onnxruntime
)
