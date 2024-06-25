
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" 
							   "${CMAKE_CURRENT_SOURCE_DIR}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" 
							   "${CMAKE_CURRENT_SOURCE_DIR}/*.idl")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCE_FILES})

SET(NVCODEC_HEADER "../extern/NvidiaCodec/Include/cuviddec.h" "../extern/NvidiaCodec/Include/nvcuvid.h" "../extern/NvidiaCodec/Include/nvEncodeAPI.h")

source_group("NVCodec" FILES ${NVCODEC_HEADER})

set(VERTEX_SHADERS DX12Lib/DXWrapper/Shaders/Basic_VS.hlsl DX12Lib/DXWrapper/Shaders/Depth_VS.hlsl)

set(PIXEL_SHADERS 
	DX12Lib/DXWrapper/Shaders/Basic_PS.hlsl 
 	DX12Lib/DXWrapper/Shaders/BasicPBR_PS.hlsl 
	DX12Lib/DXWrapper/Shaders/AlphaTest_PS.hlsl
	DX12Lib/DXWrapper/Shaders/AlphaTestPBR_PS.hlsl
	DX12Lib/DXWrapper/Shaders/AlphaTestDepth_PS.hlsl)

set(INCLUDE_SHADERS DX12Lib/DXWrapper/Shaders/Default.hlsli DX12Lib/DXWrapper/Shaders/Common.hlsli DX12Lib/DXWrapper/Shaders/LightingUtil.hlsli)

set(SHADERS ${VERTEX_SHADERS} ${PIXEL_SHADERS} ${INCLUDE_SHADERS})

source_group("DX12Lib/DXWrapper/Shaders" FILES ${SHADERS})

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

add_library(DirectXLib STATIC ${SOURCE_FILES} ${NVCODEC_HEADER} ${SHADERS})
target_precompile_headers(DirectXLib PRIVATE "DX12Lib/pch.h")

set_target_properties(DirectXLib PROPERTIES LINKER_LANGUAGE CXX CXX_STANDARD 20)

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")


target_include_directories(DirectXLib 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/directx 
	PUBLIC "../extern/NVidiaCodec/Include" "../extern/FFmpeg/include"
	PUBLIC "../extern/PIX/Include")

target_link_libraries(DirectXLib 
    PUBLIC assimp
    PUBLIC DirectXTex
    PUBLIC DirectXTK12
    PUBLIC spdlog
    PUBLIC ws2_32.lib
    PUBLIC "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avcodec.lib"
    PUBLIC "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avformat.lib"
    PUBLIC "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avutil.lib"
    PUBLIC "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/swresample.lib"
    PUBLIC "${PROJECT_SOURCE_DIR}/extern/NvidiaCodec/Lib/nvcuvid.lib"
    PUBLIC "${PROJECT_SOURCE_DIR}/extern/NvidiaCodec/Lib/nvencodeapi.lib"
    PUBLIC "${PROJECT_SOURCE_DIR}/extern/PIX/Lib/WinPixEventRuntime.lib"
    PUBLIC d3d12.lib
    PUBLIC dxgi.lib
    PUBLIC dxguid.lib)

file(GLOB DLL_FILES "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/*.dll")