
set(VERTEX_SHADERS Shaders/Voxel_VS.hlsl Shaders/VoxelDisplay_VS.hlsl)
set(GEOMETRY_SHADERS Shaders/Voxel_GS.hlsl Shaders/VoxelDisplay_GS.hlsl)
set(PIXEL_SHADERS Shaders/Voxel_PS.hlsl Shaders/VoxelDisplay_PS.hlsl)

set(INCLUDE_SHADERS 
	Shaders/VoxelUtils.hlsli 
	Shaders/TechniquesCompat.h
)

set(RAYTRACING_SHADERS
	Shaders/RaytracingUtils.hlsli
	Shaders/Raytracing.hlsl
	Shaders/RaytracingShadow.hlsl
)

set(COMPUTE_SHADERS 
	Shaders/PrefixSum_CS.hlsl
	Shaders/ClearBufferShader_CS.hlsl
	Shaders/ClusterReduce_CS.hlsl
	Shaders/ComputeAABB_CS.hlsl
	Shaders/ComputeNeighbour_CS.hlsl
	Shaders/FacePenalty_CS.hlsl
	Shaders/FastSlic_CS.hlsl
	Shaders/GaussianFilter_CS.hlsl
	Shaders/LerpRadiance_CS.hlsl
	Shaders/LightTransport_CS.hlsl
	Shaders/LightTransportDepth_CS.hlsl
	Shaders/LightTransportIndirect_CS.hlsl
	Shaders/LitVoxels_CS.hlsl)

set(SHADERS ${VERTEX_SHADERS} ${GEOMETRY_SHADERS} ${PIXEL_SHADERS} ${INCLUDE_SHADERS} ${COMPUTE_SHADERS} ${RAYTRACING_SHADERS})

source_group("Shaders" FILES ${SHADERS})

set_source_files_properties( ${VERTEX_SHADERS} ${GEOMETRY_SHADERS} ${PIXEL_SHADERS} ${COMPUTE_SHADERS} ${RAYTRACING_SHADERS}
    PROPERTIES
        VS_SHADER_OBJECT_FILE_NAME "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders/%(Filename).cso"
        VS_SHADER_OUTPUT_HEADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders/Include/%(Filename).h"
		VS_SHADER_VARIABLE_NAME "g_p%(Filename)"
        VS_SHADER_MODEL 6.5
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

set_source_files_properties( ${RAYTRACING_SHADERS}
	PROPERTIES 
	VS_SHADER_TYPE Library
)

set(TECHNIQUES
	Techniques/Technique.h
	Techniques/Technique.cpp
	Techniques/BuildAABBsTechnique.h
	Techniques/BuildAABBsTechnique.cpp
	Techniques/ClusterVisibility.h
	Techniques/ClusterVisibility.cpp
	Techniques/ClusterVoxels.h
	Techniques/ClusterVoxels.cpp
	Techniques/ComputeNeighboursTechnique.h
	Techniques/ComputeNeighboursTechnique.cpp
	Techniques/DisplayVoxelScene.h
	Techniques/DisplayVoxelScene.cpp
	Techniques/FacePenaltyTechnique.h
	Techniques/FacePenaltyTechnique.cpp
	Techniques/GaussianFilterTechnique.h
	Techniques/GaussianFilterTechnique.cpp
	Techniques/LerpRadianceTechnique.h
	Techniques/LerpRadianceTechnique.cpp
	Techniques/LightTransportTechnique.h
	Techniques/LightTransportTechnique.cpp
	Techniques/LightVoxel.h
	Techniques/LightVoxel.cpp
	Techniques/MergeClusters.h
	Techniques/MergeClusters.cpp
	Techniques/PrefixSumVoxels.h
	Techniques/PrefixSumVoxels.cpp
	Techniques/SceneDepthTechnique.h
	Techniques/SceneDepthTechnique.cpp
	Techniques/VoxelizeScene.h
	Techniques/VoxelizeScene.cpp)

source_group("Techniques" FILES ${TECHNIQUES})

set(CVGI_HELPERS
	Helpers/BufferManager.h
	Helpers/BufferManager.cpp
	Helpers/CVGIDataTypes.h
	Helpers/CVGIDataTypes.cpp
	Helpers/RayTracingHelpers.h
	Helpers/RayTracingHelpers.cpp
	Helpers/RaytracingStateObject.h
	Helpers/RaytracingStateObject.cpp)

source_group("Helpers" FILES ${CVGI_HELPERS})

set(SOURCE_FILES
	ClusteredVoxelGI.h
	ClusteredVoxelGI.cpp
	VoxelScene.h
	VoxelScene.cpp
	CameraController.h
	CameraController.cpp
	VoxelCamera.h
	VoxelCamera.cpp
)

add_executable(ClusteredVoxelGI WIN32 ${SOURCE_FILES} ${TECHNIQUES} ${CVGI_HELPERS} ${SHADERS})

set_property(TARGET ClusteredVoxelGI PROPERTY CXX_STANDARD 20)

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(ClusteredVoxelGI PRIVATE DirectXLib)
target_link_libraries(ClusteredVoxelGI PRIVATE d3d12.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE dxgi.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE D3DCompiler.lib)
target_link_libraries(ClusteredVoxelGI PRIVATE DirectXTK12)