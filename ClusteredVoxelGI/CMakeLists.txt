set(VERTEX_SHADERS )
set(PIXEL_SHADERS )
set(INCLUDE_SHADERS )

set(SHADERS ${VERTEX_SHADERS} ${PIXEL_SHADERS} ${INCLUDE_SHADERS})

source_group("Shaders" FILES ${SHADERS})

set_source_files_properties( ${VERTEX_SHADERS} ${PIXEL_SHADERS}
    PROPERTIES
        VS_SHADER_OBJECT_FILE_NAME "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders/%(Filename).cso"
        VS_SHADER_MODEL 5.1
        VS_SHADER_DISABLE_OPTIMIZATIONS $<$<CONFIG:Debug>:ON>
        VS_SHADER_ENABLE_DEBUG $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:ON>
)

set_source_files_properties( ${VERTEX_SHADERS}
    PROPERTIES 
        VS_SHADER_TYPE Vertex
	VS_SHADER_ENTRYPOINT VS
)

set_source_files_properties( ${PIXEL_SHADERS}
    PROPERTIES 
        VS_SHADER_TYPE Pixel
	VS_SHADER_ENTRYPOINT PS
)


set(SOURCE_FILES
	ClusteredVoxelGI.cpp
	VoxelScene.h
	VoxelScene.cpp
	CameraController.h
	CameraController.cpp
)

add_executable(ClusteredVoxelGI WIN32 ${SOURCE_FILES} ${SHADERS})

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(ClusteredVoxelGI PRIVATE DirectXLib)
target_link_libraries(ClusteredVoxelGI PRIVATE d3d12.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE dxgi.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE D3DCompiler.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE DirectXTK12)