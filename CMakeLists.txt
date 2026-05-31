cmake_minimum_required(VERSION 3.20)
project(nori-dxr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_definitions(-DUNICODE -D_UNICODE)

set(NORI_EXT ${CMAKE_SOURCE_DIR}/ext)

# build dependencies

# PugiXML
add_library(pugixml STATIC ${NORI_EXT}/pugixml/src/pugixml.cpp)
target_include_directories(pugixml PUBLIC ${NORI_EXT}/pugixml/src)

# TBB
set(TBB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_STATIC ON  CACHE BOOL "" FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${NORI_EXT}/tbb ${CMAKE_BINARY_DIR}/tbb_build)

# include directories
set(NORI_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${NORI_EXT}/eigen
    ${NORI_EXT}/pugixml/src
    ${NORI_EXT}/tbb/include
    ${NORI_EXT}/pcg32
    ${NORI_EXT}/tinyformat
    ${NORI_EXT}/filesystem
    ${NORI_EXT}/hypothesis
    ${NORI_EXT}/nanogui/ext/nanovg/example
    ${NORI_EXT}/nanogui/ext/nanovg/src
    ${NORI_EXT}/openexr/IlmBase/Imath
    ${NORI_EXT}/openexr/IlmBase/Iex
    ${NORI_EXT}/openexr/IlmBase/Half
)

set(NORI_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cpu/common.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/object.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/proplist.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/parser.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/mesh.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/obj.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/scene.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/accel.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/perspective.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/independent.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/warp.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/rfilter.cpp
    # BSDFs
    ${CMAKE_SOURCE_DIR}/src/cpu/diffuse.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/mirror.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/dielectric.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/microfacet.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/disney.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/hair.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/homogeneous.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/heterogeneous.cpp
)

# dxc shader compiler
find_program(DXC_EXECUTABLE dxc
    PATHS "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
          "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
          "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22000.0/x64"
)
if(NOT DXC_EXECUTABLE)
    message(FATAL_ERROR "dxc.exe not found")
endif()
message(STATUS "Found DXC: ${DXC_EXECUTABLE}")

set(SHADER_SRC ${CMAKE_SOURCE_DIR}/shaders/Shaders.hlsl)
set(SHADER_VOL ${CMAKE_SOURCE_DIR}/shaders/Volume.hlsl)
set(SHADER_CSO ${CMAKE_BINARY_DIR}/Shaders.cso)
add_custom_command(
    OUTPUT ${SHADER_CSO}
    COMMAND ${DXC_EXECUTABLE} -T lib_6_3 -Fo ${SHADER_CSO} ${SHADER_SRC}
    DEPENDS ${SHADER_SRC} ${SHADER_VOL}
    COMMENT "Compiling Shaders.hlsl"
    VERBATIM
)
add_custom_target(compile_shaders DEPENDS ${SHADER_CSO})

add_executable(nori-dxr
    ${CMAKE_SOURCE_DIR}/src/gpu/main.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu/Win32Application.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu/DXRApp.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu/DXRApp_Device.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu/DXRApp_Scene.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu/DXRApp_Textures.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu/DXRApp_Pipeline.cpp
    ${CMAKE_SOURCE_DIR}/src/gpu/DXRApp_Camera.cpp
    # Emitters
    ${CMAKE_SOURCE_DIR}/src/cpu/area.cpp
    ${NORI_SOURCES}
)
add_dependencies(nori-dxr compile_shaders)

target_include_directories(nori-dxr PRIVATE ${NORI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/gpu)

if(MSVC)
    target_compile_definitions(nori-dxr PRIVATE
        _CRT_SECURE_NO_WARNINGS
        __TBB_NO_IMPLICIT_LINKAGE
        NOMINMAX
    )
    target_compile_options(nori-dxr PRIVATE /O2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(nori-dxr PRIVATE -O3)
endif()

target_link_libraries(nori-dxr PRIVATE
    d3d12 dxgi dxguid d3dcompiler
    pugixml tbb_static
)

add_custom_command(TARGET nori-dxr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_CSO} $<TARGET_FILE_DIR:nori-dxr>/Shaders.cso
)