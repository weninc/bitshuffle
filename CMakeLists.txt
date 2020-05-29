cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(bitshuffle)

find_package(HDF5 REQUIRED)
if(LINUX) 
    add_compile_options(-mavx2)
    add_compile_options(-ffast-math)
endif()


add_library(bshuf STATIC
    src/bitshuffle.c
    src/bitshuffle_core.c
    src/iochain.c
    lz4/lz4.c
)

target_include_directories(bshuf PRIVATE
    lz4
)

target_include_directories(bshuf PUBLIC
    $<INSTALL_INTERFACE:include>
)

add_library(h5bshuf SHARED
    src/bshuf_h5plugin.c
    src/bshuf_h5filter.c
)

target_include_directories(h5bshuf PRIVATE
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(h5bshuf
    ${HDF5_LIBRARIES}
    bshuf
)

install(FILES
    src/bitshuffle.h
    src/bitshuffle_core.h
    DESTINATION include
)

install(TARGETS bshuf
    EXPORT bitshuffleTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(TARGETS h5bshuf
    LIBRARY DESTINATION lib/hdf5/plugin
)

install(EXPORT bitshuffleTargets
    FILE bitshuffleTargets.cmake
    NAMESPACE bitshuffle::
    DESTINATION lib/cmake/bitshuffle
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/bitshuffleConfig.cmake.in"
    ${CMAKE_CURRENT_BINARY_DIR}/bitshuffleConfig.cmake
    INSTALL_DESTINATION lib/cmake/bitshuffle
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/bitshuffleConfig.cmake
    DESTINATION lib/cmake/bitshuffle
)
