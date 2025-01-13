cmake_minimum_required(VERSION 3.29.1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)



add_compile_definitions(_UNICODE UNICODE SPDLOG_WCHAR_TO_UTF8_SUPPORT)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Output directory for static lib (.LIB)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Output directory for shared lib (.DLL)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Output directory for executables (.EXE)

project(DistributedRendering C CXX CUDA)

message("Dir is: " ${CMAKE_BINARY_DIR})

# PATH variable for res files
SET(ASSETS_DIR ${PROJECT_SOURCE_DIR}/Assets/)

SET( ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "Build Assimp Tools" FORCE )
SET( ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "Build Assimp Samples" FORCE )
SET( ASSIMP_BUILD_TESTS OFF CACHE BOOL "Build Assimp Tests" FORCE )

add_subdirectory( extern/assimp )

set_target_properties( assimp uninstall UpdateAssimpLibsDebugSymbolsAndDLLs zlibstatic 
    PROPERTIES
        FOLDER assimp 
)

add_subdirectory(extern/DirectXTex)

set_target_properties (DirectXTex texassemble texconv texdiag ddsview
	PROPERTIES
		FOLDER DirectXTex
)

set(ZSTD_BUILD_TESTS OFF CACHE BOOL "Build Zstd tests" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "Disable Zstd legacy support" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "Disable Zstd legacy support" FORCE)

add_subdirectory(extern/zstd/build/cmake)

set_target_properties( zstd clean-all libzstd_static zstd
    PROPERTIES
         FOLDER zstd 
 )

set(ENET_STATIC 1)


add_subdirectory(extern/spdlog)
add_subdirectory(extern/DirectXTK12)
add_subdirectory(extern/glfw)
add_subdirectory(extern/enet)
add_subdirectory(extern/imgui)
add_subdirectory(DirectXLib)
add_subdirectory(ClusteredVoxelGI)
add_subdirectory(LocalIllumination)
add_subdirectory(StreamingClient)

set_directory_properties( PROPERTIES VS_STARTUP_PROJECT ClusteredVoxelGI)
