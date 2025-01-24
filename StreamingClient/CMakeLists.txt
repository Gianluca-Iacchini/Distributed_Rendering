set( SRC_FILES 
	main.cpp 
	Helpers.h 
	Helpers.cpp 
	FFmpegDemuxer.h 
	FFmpegDemuxer.cpp 
	NVDecoder.h 
	NVDecoder.cpp 
	StreamRenderer.h
	StreamRenderer.cpp
	ColorSpace.h
	../extern/glad/src/glad.c 
	../extern/glad/include/glad/glad.h)

set( SHADERS Shaders/BasicVertex.vs Shaders/BasicFragment.fs)

set (CUDA_FILES ColorSpace.cu)

source_group("Shaders" FILES ${SHADERS})

source_group("imGui" FILES ${IMGUI_SOURCE_GL})

set(COMMONS_SOURCE 
	"../Commons/NetworkManager.h"
	"../Commons/NetworkManager.cpp"
	"../Commons/UIHelpers.h"
	"../Commons/UIHelpers.cpp"
)

set(COMMONS_SOURCE 
	"../Commons/NetworkManager.h"
	"../Commons/NetworkManager.cpp"
	"../Commons/UIHelpers.h"
	"../Commons/UIHelpers.cpp"
)

source_group("Commons" FILES ${COMMONS_SOURCE})

find_package( OpenGL REQUIRED )
find_package( CUDAToolkit REQUIRED)


include_directories( ${OPENGL_INCLUDE_DIRS} 
	../extern/glad/include/
	../extern/FFmpeg/include
	${IMGUI_INCLUDE}
	../extern/NvidiaCodec/Include
	../extern/enet/include
	../Commons/
	${zstd_SOURCE_DIR}/lib)


add_executable(StreamingClient ${SRC_FILES} ${IMGUI_SOURCE_GL} ${COMMONS_SOURCE} ${SHADERS} ${CUDA_FILES})

target_link_libraries(StreamingClient
	PRIVATE glfw
	PRIVATE spdlog
    	PRIVATE ws2_32.lib
    	PRIVATE winmm.lib
        PRIVATE libzstd_static
    	PRIVATE "${PROJECT_SOURCE_DIR}/extern/enet/enet64.lib"
	PRIVATE CUDA::cudart
	PRIVATE CUDA::cuda_driver

	PRIVATE "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avcodec.lib"
	PRIVATE "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avformat.lib"
	PRIVATE "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avutil.lib"
	PRIVATE "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/swresample.lib"
	PRIVATE "${PROJECT_SOURCE_DIR}/extern/NvidiaCodec/Lib/nvcuvid.lib"
	)

set_property(TARGET ClusteredVoxelGI PROPERTY CXX_STANDARD 20)

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}" -DCVGI_GL=1)
file(GLOB DLL_FILES "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/*.dll")

foreach(DLL ${DLL_FILES})
	add_custom_command(TARGET StreamingClient POST_BUILD
    		COMMAND ${CMAKE_COMMAND} -E copy_if_different
        		${DLL} $<TARGET_FILE_DIR:StreamingClient>)
endforeach()