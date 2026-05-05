cmake_minimum_required(VERSION 3.20)
project(nori-dxr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DUNICODE -D_UNICODE)

set(NORI_ROOT ${CMAKE_SOURCE_DIR}/..)
set(NORI_EXT  ${NORI_ROOT}/ext)

#build dependencies 

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
    ${NORI_ROOT}/include
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
    ${NORI_ROOT}/src/common.cpp
    ${NORI_ROOT}/src/object.cpp
    ${NORI_ROOT}/src/proplist.cpp
    ${NORI_ROOT}/src/parser.cpp
    ${NORI_ROOT}/src/mesh.cpp
    ${NORI_ROOT}/src/obj.cpp
    ${NORI_ROOT}/src/scene.cpp
    ${NORI_ROOT}/src/accel.cpp
    ${NORI_ROOT}/src/perspective.cpp
    ${NORI_ROOT}/src/independent.cpp
    ${NORI_ROOT}/src/rfilter.cpp
    ${NORI_ROOT}/src/warp.cpp
    # BSDFs
    ${NORI_ROOT}/src/diffuse.cpp
    ${NORI_ROOT}/src/mirror.cpp
    ${NORI_ROOT}/src/dielectric.cpp
    ${NORI_ROOT}/src/microfacet.cpp
    ${NORI_ROOT}/src/disney.cpp
    ${NORI_ROOT}/src/hair.cpp
    ${NORI_ROOT}/src/homogeneous.cpp
    ${NORI_ROOT}/src/heterogeneous.cpp
    # Emitters
    ${NORI_ROOT}/src/area.cpp
    # Integrators
    ${NORI_ROOT}/src/normals.cpp
    ${NORI_ROOT}/src/simple.cpp
    ${NORI_ROOT}/src/ao.cpp
    ${NORI_ROOT}/src/direct.cpp
    ${NORI_ROOT}/src/path_mats.cpp
    ${NORI_ROOT}/src/path_ems.cpp
    ${NORI_ROOT}/src/path_mis.cpp
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
    src/main.cpp
    src/Win32Application.cpp
    src/DXRApp.cpp
    ${NORI_SOURCES}
)
add_dependencies(nori-dxr compile_shaders)

target_include_directories(nori-dxr PRIVATE ${NORI_INCLUDE_DIRS})

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