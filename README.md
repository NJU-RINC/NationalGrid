# NationalGrid

Version 1 全流程示意图

![svg](https://github.com/NJU-RINC/NationalGrid/blob/main/resource/image/v1.svg)


cpp 编译命令
```
g++ xxx.cpp -o xxx `pkg-config --cflags --libs opencv4` -I/usr/local/cuda-10.2/targets/aarch64-linux/include
```

libtorch 编译命令
```
cd research/cpp

rm -rf build

mkdir build

cd build

cmake .. \
-DCMAKE_PREFIX_PATH="/usr/local/lib/python3.6/dist-packages/torch/share/cmake"

make -j6
```