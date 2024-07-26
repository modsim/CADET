call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
cd build
set MKLROOT="C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
cmake -DCMAKE_INSTALL_PREFIX=..\out\install\aRELEASE -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows-static -DENABLE_STATIC_LINK_LAPACK=ON -DENABLE_STATIC_LINK_DEPS=ON -DBLA_VENDOR=Intel10_64lp_seq --fresh ../
msbuild.exe INSTALL.vcxproj /p:Configuration=Release;Platform=x64
