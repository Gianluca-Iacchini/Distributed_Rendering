set(SRC_FILES main.cpp Helpers.h Helpers.cpp ../extern/glad/src/glad.c ../extern/glad/include/glad/glad.h)

find_package( OpenGL REQUIRED )
find_package( CUDAToolkit REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS} ../extern/glad/include/)


add_executable(StreamingClient ${SRC_FILES})

target_link_libraries(StreamingClient
	PRIVATE glfw
	PRIVATE spdlog
	PRIVATE CUDA::cudart
	PRIVATE CUDA::cuda_driver
	)


add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

