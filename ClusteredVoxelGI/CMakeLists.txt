set(VERTEX_SHADERS Shaders/Voxel_VS.hlsl Shaders/VoxelDisplay_VS.hlsl)
set(GEOMETRY_SHADERS Shaders/Voxel_GS.hlsl Shaders/VoxelDisplay_GS.hlsl)
set(PIXEL_SHADERS Shaders/Voxel_PS.hlsl Shaders/VoxelDisplay_PS.hlsl)
set(COMPUTE_SHADERS Shaders/PrefixSum_CS.hlsl)
set(INCLUDE_SHADERS Shaders/VoxelUtils.hlsli)

set(SHADERS ${VERTEX_SHADERS} ${GEOMETRY_SHADERS} ${PIXEL_SHADERS} ${INCLUDE_SHADERS} ${COMPUTE_SHADERS})

source_group("Shaders" FILES ${SHADERS})

set_source_files_properties( ${VERTEX_SHADERS} ${GEOMETRY_SHADERS} ${PIXEL_SHADERS} ${COMPUTE_SHADERS}
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

set_source_files_properties( ${GEOMETRY_SHADERS}
    PROPERTIES 
        VS_SHADER_TYPE Geometry
	VS_SHADER_ENTRYPOINT GS
)

set_source_files_properties( ${PIXEL_SHADERS}
    PROPERTIES 
        VS_SHADER_TYPE Pixel
	VS_SHADER_ENTRYPOINT PS
)

set_source_files_properties( ${COMPUTE_SHADERS}
    PROPERTIES 
        VS_SHADER_TYPE Compute
	VS_SHADER_ENTRYPOINT CS
)

set(SOURCE_FILES
	CVGIDataTypes.h
	ClusteredVoxelGI.h
	ClusteredVoxelGI.cpp
	VoxelScene.h
	VoxelScene.cpp
	VoxelMaterial.h
	VoxelMaterial.cpp
	CameraController.h
	CameraController.cpp
	VoxelCamera.h
	VoxelCamera.cpp
	VoxelBufferManager.h
	VoxelBufferManager.cpp
)

add_executable(ClusteredVoxelGI WIN32 ${SOURCE_FILES} ${SHADERS})

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(ClusteredVoxelGI PRIVATE DirectXLib)
target_link_libraries(ClusteredVoxelGI PRIVATE d3d12.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE dxgi.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE D3DCompiler.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE DirectXTK12)