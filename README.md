# TensorRT7-DCNv2-Plugin
基于TensorRT7实现DCNv2插件

## update
**2021-10-27 support Tensorrt8**

## prerequirements
1. CUB-1.8.0
2. TensorRT-7.0.0.11

## dependencies
1. TensorRT 7.0
2. onnx-tensorrt 7.0

1. clone TensorRT release/7.0版本
2. 将DCNv2文件夹、InferPlugin.cpp以及CMakeLists拷贝到TensorRT/plugin中
3. 在TensorRT目录创建build目录，进入后运行
```
cmake .. -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF -DTRT_LIB_DIR=$TENSORRT_LIB_PATH -DTRT_BIN_DIR=`pwd`/out
-DBUILD_PLUGINS=ON -DCUB_ROOT_DIR=$CUB_PATH
make -j4
```
4. 编译完成后会生成libnvinfer_plugin库
5. 将builtin_op_importer.cpp拷贝到onnx-tensorrt中编译libnvonnxparser库

DCNv2的实现代码摘自CaoWGG/TensorRT-CenterNet
