# TensorRT7-DCNv2-Plugin
DCNv2 plugin implemented on TensorRT7

## update
**2021-10-27 support Tensorrt8 （Since TensorRT8 make many method noexcept，so you have to add noexcept to the method）**

## prerequirements
1. CUB-1.8.0
2. TensorRT-7.0.0.11

## dependencies
1. TensorRT 7.0
2. onnx-tensorrt 7.0

1. clone TensorRT release/7.0
2. put DCNv2 directory、InferPlugin.cpp and CMakeLists to TensorRT/plugin
3. 
```
cd TensorRT
cmake .. -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF -DTRT_LIB_DIR=$TENSORRT_LIB_PATH -DTRT_BIN_DIR=`pwd`/out
-DBUILD_PLUGINS=ON -DCUB_ROOT_DIR=$CUB_PATH
make -j4
```
4. after compile you will have libnvinfer_plugin.so with DCNv2
5. put builtin_op_importer.cpp to onnx-tensorrt and compile onnx-tensorrt to get libnvonnxparser.so
6. use libnvinfer_plugin.so and libnvonnxparser.so replace the origin so in TensorRT/lib

code of DCNv2 come from CaoWGG/TensorRT-CenterNet
