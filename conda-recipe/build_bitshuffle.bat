mkdir build
cd build

cmake -G"NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%" ^
    -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
    -DSNAPPY_INCLUDE_DIRS="%LIBRARY_INC%" ^
    ..

cmake --build . --config Release
cmake --build . --target install
