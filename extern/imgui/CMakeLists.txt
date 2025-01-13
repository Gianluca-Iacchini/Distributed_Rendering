set(IMGUI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# Include all .h and .cpp files directly in the current folder (subfolder)
file(GLOB IMGUI_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Include all files from the "backends" directory
set(BACKENDS_FILES 
	${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_dx12.h
	${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_dx12.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_win32.h
	${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_win32.cpp)

# Include all files from the "misc" directory
file(GLOB MISC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/misc/*)

# Combine all files into a single variable
set(IMGUI_SOURCE ${IMGUI_FILES} ${BACKENDS_FILES} ${MISC_FILES} PARENT_SCOPE)