cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(bitshuffle)

find_package(HDF5 static REQUIRED)

set(WINDOWS_EXPORT_ALL_SYMBOLS True)

add_library(h5bshuf SHARED
    src/bitshuffle.c
    src/bitshuffle_core.c
    src/iochain.c
    lz4/lz4.c
    src/bshuf_h5plugin.c
    src/bshuf_h5filter.c
)

target_include_directories(h5bshuf PRIVATE
    lz4
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(h5bshuf
    ${HDF5_LIBRARIES}
)

install(TARGETS h5bshuf
    LIBRARY DESTINATION lib/hdf5/plugin
)


