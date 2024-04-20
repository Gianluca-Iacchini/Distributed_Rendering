add_executable(LocalIllumination WIN32 main.cpp FrameResource.h FrameResource.cpp)

add_definitions(-DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(LocalIllumination PRIVATE DirectXLib)
target_link_libraries(LocalIllumination PRIVATE d3d12.lib)
target_link_libraries(LocalIllumination PRIVATE dxgi.lib)
target_link_libraries(LocalIllumination PRIVATE D3DCompiler.lib)
target_link_libraries(LocalIllumination PRIVATE DirectXTK12)

target_include_directories(DirectXLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})