
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" 
							   "${CMAKE_CURRENT_SOURCE_DIR}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" 
							   "${CMAKE_CURRENT_SOURCE_DIR}/*.idl")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCE_FILES})

add_library(DirectXLib STATIC "${SOURCE_FILES}")

set_target_properties(DirectXLib PROPERTIES LINKER_LANGUAGE CXX)



target_include_directories(DirectXLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(DirectXLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/directx)
target_link_libraries(DirectXLib 
    PUBLIC DirectXTK12
    PUBLIC d3d12.lib
    PUBLIC dxgi.lib
    PUBLIC dxguid.lib)