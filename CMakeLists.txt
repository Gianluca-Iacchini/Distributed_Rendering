cmake_minimum_required(VERSION 3.29.1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


add_compile_definitions(_UNICODE UNICODE)

project(DistributedRendering)

# PATH variable for res files
SET(ASSETS_DIR ${PROJECT_SOURCE_DIR}/Assets/)


add_subdirectory(DirectXTex)
add_subdirectory(DirectXTK12)
add_subdirectory(DirectXLib)
add_subdirectory(LocalIllumination)

set_directory_properties( PROPERTIES VS_STARTUP_PROJECT LocalIllumination)
