# CMakeLists.txt - Cấu hình biên dịch cho VMS C++ Server Core
cmake_minimum_required(VERSION 3.12)
project(vms_server_cpp CXX)

# Thiết lập chuẩn C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── 1. TÌM KIẾM CÁC THƯ VIỆN BÊN NGOÀI ───

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(Threads REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# libpqxx & PostgreSQL (Dành cho DB PostgreSQL)
find_package(PostgreSQL REQUIRED)
# Thư viện C++ wrapper libpqxx
find_path(PQXX_INCLUDE_DIR NAMES pqxx/pqxx INDEX_DIRS /usr/include /usr/local/include)
find_library(PQXX_LIBRARY NAMES pqxx INDEX_DIRS /usr/lib /usr/local/lib)

if(NOT PQXX_INCLUDE_DIR OR NOT PQXX_LIBRARY)
    message(STATUS "pqxx not found via standard search. If building from source, make sure to set PQXX_INCLUDE_DIR and PQXX_LIBRARY manually.")
else()
    message(STATUS "libpqxx found:")
    message(STATUS "    includes: ${PQXX_INCLUDE_DIR}")
    message(STATUS "    library: ${PQXX_LIBRARY}")
endif()

# ONNX Runtime (C/C++ API)
# Keep Debug/Release import libraries separate so the MSVC generator does not
# accidentally link Release builds through vcpkg's debug tree.
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime)
if(EXISTS "$ENV{ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_LIBRARY_RELEASE "$ENV{ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
elseif(EXISTS "D:/vcpkg/installed/x64-windows/lib/onnxruntime.lib")
    set(ONNXRUNTIME_LIBRARY_RELEASE "D:/vcpkg/installed/x64-windows/lib/onnxruntime.lib")
else()
    find_library(ONNXRUNTIME_LIBRARY_RELEASE NAMES onnxruntime PATH_SUFFIXES lib)
endif()
if(EXISTS "$ENV{ONNXRUNTIME_DIR}/debug/lib/onnxruntime.lib")
    set(ONNXRUNTIME_LIBRARY_DEBUG "$ENV{ONNXRUNTIME_DIR}/debug/lib/onnxruntime.lib")
elseif(EXISTS "D:/vcpkg/installed/x64-windows/debug/lib/onnxruntime.lib")
    set(ONNXRUNTIME_LIBRARY_DEBUG "D:/vcpkg/installed/x64-windows/debug/lib/onnxruntime.lib")
else()
    find_library(ONNXRUNTIME_LIBRARY_DEBUG NAMES onnxruntime PATH_SUFFIXES debug/lib)
endif()

if(ONNXRUNTIME_LIBRARY_RELEASE AND ONNXRUNTIME_LIBRARY_DEBUG)
    set(ONNXRUNTIME_LIBRARIES optimized ${ONNXRUNTIME_LIBRARY_RELEASE} debug ${ONNXRUNTIME_LIBRARY_DEBUG})
elseif(ONNXRUNTIME_LIBRARY_RELEASE)
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY_RELEASE})
else()
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARIES})
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARIES)
    message(WARNING "ONNX Runtime not found! Please set ONNXRUNTIME_DIR environment variable or specify paths manually.")
else()
    message(STATUS "ONNX Runtime found:")
    message(STATUS "    includes: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "    release library: ${ONNXRUNTIME_LIBRARY_RELEASE}")
    message(STATUS "    debug library: ${ONNXRUNTIME_LIBRARY_DEBUG}")
endif()

# ─── 2. THIẾT LẬP DỰ ÁN ───

# Đăng ký thư mục Header
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PostgreSQL_INCLUDE_DIRS}
    ${PQXX_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Danh sách mã nguồn
set(SOURCE_FILES
    src/main.cpp
    src/InferenceEngine.cpp
    src/StorageEngine.cpp
    src/AnprIntegration.cpp
)

# Tạo tệp thực thi (executable)
add_executable(vms_server_core ${SOURCE_FILES})

target_compile_definitions(vms_server_core PRIVATE NOMINMAX)

# Keep the cameraAI3 ANPR assets beside the executable so relative runtime
# paths work from both the source tree and Visual Studio's Debug/Release dirs.
add_custom_command(TARGET vms_server_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vms_server_core>/models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/models/best_plateSegment.onnx"
        "$<TARGET_FILE_DIR:vms_server_core>/models/best_plateSegment.onnx"
    COMMENT "Copying cameraAI3 plate model to the runtime directory"
)

option(VMS_ENABLE_ASAN "Enable AddressSanitizer for the C++ service" OFF)
if(VMS_ENABLE_ASAN)
    if(MSVC)
        target_compile_options(vms_server_core PRIVATE /fsanitize=address)
    else()
        target_compile_options(vms_server_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(vms_server_core PRIVATE -fsanitize=address,undefined)
    endif()
endif()

# Liên kết các thư viện
target_link_libraries(vms_server_core
    PRIVATE
    ${OpenCV_LIBS}
    ${PostgreSQL_LIBRARIES}
    ${PQXX_LIBRARY}
    ${ONNXRUNTIME_LIBRARIES}
    Threads::Threads
    cppzmq
)

# Thiết lập tùy chọn biên dịch bổ sung tùy theo Compiler
if(MSVC)
    target_compile_options(vms_server_core PRIVATE /W4 /EHsc /bigobj)
else()
    target_compile_options(vms_server_core PRIVATE -Wall -Wextra -O3)
endif()
