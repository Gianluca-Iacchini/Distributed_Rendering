set(VERTEX_SHADERS )
set(PIXEL_SHADERS )

set(INCLUDE_SHADERS 
	../VoxelUtils/Shaders/VoxelUtils.hlsli 
	../VoxelUtils/Shaders/TechniquesCompat.h
)

set(RAYTRACING_SHADERS
	../VoxelUtils/Shaders/RaytracingUtils.hlsli
	../VoxelUtils/Shaders/Raytracing.hlsl
	../VoxelUtils/Shaders/RaytracingShadow.hlsl
)

set(COMPUTE_SHADERS 

	../VoxelUtils/Shaders/ClearBufferShader_CS.hlsl
	../VoxelUtils/Shaders/GaussianFilter_CS.hlsl
	../VoxelUtils/Shaders/LightTransportDepth_CS.hlsl
	../VoxelUtils/Shaders/LightTransportIndirect_CS.hlsl

	Shaders/RadianceFromNetwork_CS.hlsl
	Shaders/GaussianOnly_CS.hlsl
)

source_group("Shaders" FILES ${SHADERS})

set_source_files_properties( ${VERTEX_SHADERS} ${GEOMETRY_SHADERS} ${PIXEL_SHADERS} ${COMPUTE_SHADERS} ${RAYTRACING_SHADERS}
    PROPERTIES

        VS_SHADER_OBJECT_FILE_NAME "%(RelativeDir)/../Data/Shaders/%(Filename).cso"
        VS_SHADER_OUTPUT_HEADER_FILE "%(RelativeDir)/../Data/Shaders/Include/%(Filename).h"
		VS_SHADER_VARIABLE_NAME "g_p%(Filename)"
        VS_SHADER_MODEL 6.5
        VS_SHADER_DISABLE_OPTIMIZATIONS $<$<CONFIG:Debug>:ON>
        VS_SHADER_ENABLE_DEBUG $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:ON>
)


set(SHADERS ${VERTEX_SHADERS}  ${PIXEL_SHADERS} ${INCLUDE_SHADERS} ${COMPUTE_SHADERS} ${RAYTRACING_SHADERS})

source_group("Shaders" FILES ${SHADERS})


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

set_source_files_properties( ${COMPUTE_SHADERS}
    PROPERTIES 
        VS_SHADER_TYPE Compute
	VS_SHADER_ENTRYPOINT CS
)

set_source_files_properties( ${RAYTRACING_SHADERS}
	PROPERTIES 
	VS_SHADER_TYPE Library
)

set(VOX_HELPERS 
	../VoxelUtils/BufferManager.h
	../VoxelUtils/BufferManager.cpp
	../VoxelUtils/CVGIDataTypes.h
	../VoxelUtils/CVGIDataTypes.cpp
)

source_group("Helpers" FILES ${VOX_HELPERS})

set(TECHNIQUES
	../VoxelUtils/Techniques/Technique.h
	../VoxelUtils/Techniques/Technique.cpp
	../VoxelUtils/Techniques/LightTransportTechnique.h
	../VoxelUtils/Techniques/LightTransportTechnique.cpp
	../VoxelUtils/Techniques/GaussianFilterTechnique.h
	../VoxelUtils/Techniques/GaussianFilterTechnique.cpp
	../VoxelUtils/Techniques/SceneDepthTechnique.h
	../VoxelUtils/Techniques/SceneDepthTechnique.cpp
	
	RadianceFromNetworkTechnique.h
	RadianceFromNetworkTechnique.cpp
)

source_group("Techniques" FILES ${TECHNIQUES})

set(SOURCE_FILES
	LocalIllumination.h 
	LocalIllumination.cpp 
	LIScene.h
	LIScene.cpp
	CameraController.h
	CameraController.cpp
    LIUtils.h
    LIUtils.cpp
)

add_executable(LocalIllumination WIN32 ${SOURCE_FILES} ${TECHNIQUES} ${VOX_HELPERS} ${SHADERS})

set_property(TARGET LocalIllumination PROPERTY CXX_STANDARD 20)

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(LocalIllumination PRIVATE DirectXLib)
target_link_libraries(LocalIllumination PRIVATE d3d12.lib)
target_link_libraries(LocalIllumination PRIVATE dxgi.lib)
target_link_libraries(LocalIllumination PRIVATE D3DCompiler.lib)
target_link_libraries(LocalIllumination PRIVATE DirectXTK12)

target_include_directories(LocalIllumination
    PUBLIC "../VoxelUtils"
    PUBLIC "../VoxelUtils/Techniques") 

file(GLOB DLL_FILES "${PROJECT_SOURCE_DIR}/extern/PIX/Lib/*.dll")

foreach(DLL ${DLL_FILES})
	add_custom_command(TARGET LocalIllumination POST_BUILD
    		COMMAND ${CMAKE_COMMAND} -E copy_if_different
        		${DLL} $<TARGET_FILE_DIR:LocalIllumination>)
endforeach()
