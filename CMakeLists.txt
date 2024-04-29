cmake_minimum_required(VERSION 3.29.1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


add_compile_definitions(_UNICODE UNICODE)

project(DistributedRendering)

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


add_subdirectory(extern/DirectXTK12)
add_subdirectory(DirectXLib)
add_subdirectory(LocalIllumination)

set_directory_properties( PROPERTIES VS_STARTUP_PROJECT LocalIllumination)
