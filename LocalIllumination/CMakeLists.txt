set(VERTEX_SHADERS Shaders/Basic_VS.hlsl)
set(PIXEL_SHADERS Shaders/Basic_PS.hlsl)
set(INCLUDE_SHADERS Shaders/Default.hlsli Shaders/Common.hlsli Shaders/LightingUtil.hlsli)

set(SHADERS ${VERTEX_SHADERS} ${PIXEL_SHADERS} ${INCLUDE_SHADERS})

source_group("Shaders" FILES ${SHADERS})

set_source_files_properties(${SHADERS} PROPERTIES )
set_source_files_properties(${SHADERS} PROPERTIES )

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


add_executable(LocalIllumination WIN32 main.cpp FrameResource.h FrameResource.cpp ${SHADERS})

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(LocalIllumination PRIVATE DirectXLib)
target_link_libraries(LocalIllumination PRIVATE d3d12.lib)
target_link_libraries(LocalIllumination PRIVATE dxgi.lib)
target_link_libraries(LocalIllumination PRIVATE D3DCompiler.lib)
target_link_libraries(LocalIllumination PRIVATE DirectXTK12)

target_include_directories(DirectXLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})