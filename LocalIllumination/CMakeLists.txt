set(SHADERS Shaders/Basic.hlsl)

source_group("Shaders" FILES ${SHADERS})

set_source_files_properties( ${SHADERS}
    PROPERTIES
        VS_SHADER_OBJECT_FILE_NAME "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders/%(Filename).cso"
        VS_SHADER_MODEL 5.1
        VS_SHADER_DISABLE_OPTIMIZATIONS $<$<CONFIG:Debug>:ON>
        VS_SHADER_ENABLE_DEBUG $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:ON>
)

add_executable(LocalIllumination WIN32 main.cpp FrameResource.h FrameResource.cpp ${SHADERS})

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(LocalIllumination PRIVATE DirectXLib)
target_link_libraries(LocalIllumination PRIVATE d3d12.lib)
target_link_libraries(LocalIllumination PRIVATE dxgi.lib)
target_link_libraries(LocalIllumination PRIVATE D3DCompiler.lib)
target_link_libraries(LocalIllumination PRIVATE DirectXTK12)

target_include_directories(DirectXLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})