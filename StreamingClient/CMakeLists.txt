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

find_package( OpenGL REQUIRED )
find_package( CUDAToolkit REQUIRED)


include_directories( ${OPENGL_INCLUDE_DIRS} ../extern/glad/include/ ../extern/FFmpeg/include ../extern/NvidiaCodec/Include)


add_executable(StreamingClient ${SRC_FILES} ${SHADERS} ${CUDA_FILES})

target_link_libraries(StreamingClient
	PRIVATE glfw
	PRIVATE spdlog
	PRIVATE ws2_32.lib
	PRIVATE CUDA::cudart
	PRIVATE CUDA::cuda_driver

	PRIVATE "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avcodec.lib"
	PRIVATE "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avformat.lib"
	PRIVATE "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/avutil.lib"
	PRIVATE "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/swresample.lib"
	PRIVATE "${PROJECT_SOURCE_DIR}/extern/NvidiaCodec/Lib/nvcuvid.lib"
	)


add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB DLL_FILES "${PROJECT_SOURCE_DIR}/extern/FFmpeg/lib/x64/*.dll")

foreach(DLL ${DLL_FILES})
	add_custom_command(TARGET StreamingClient POST_BUILD
    		COMMAND ${CMAKE_COMMAND} -E copy_if_different
        		${DLL} $<TARGET_FILE_DIR:StreamingClient>)
endforeach()