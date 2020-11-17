//
// Created by cao on 19-12-20.
//

#include "DCNv2.hpp"
#include "dcn_v2_im2col_cuda.h"
#include <iostream>

using namespace nvinfer1;
using nvinfer1::plugin::DCNv2Plugin;
using nvinfer1::plugin::DCNv2PluginCreator;

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    if(!init[n]) {
        cublasCreate(&handle[n]);
        init[n] = 1;
    }
    return handle[n];
}

inline bool is_CHW(nvinfer1::Dims const& dims) {
    return (dims.nbDims == 3 &&
            dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
            dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
            dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

DCNv2Plugin::DCNv2Plugin(int in_channel,
                         int out_channel,
                         int kernel_H,
                         int kernel_W,
                         int deformable_group,
                         int dilation,
                         int groups,
                         int padding,
                         int stride,
                         nvinfer1::Weights const &weight, nvinfer1::Weights const &bias):_in_channel(in_channel),
                        _out_channel(out_channel),_kernel_H(kernel_H),_kernel_W(kernel_W),_deformable_group(deformable_group),
                         _dilation(dilation),_groups(groups),_padding(padding),_stride(stride),_initialized(false){

    // std::cout << "**************************** call DCNv2 construct ******************************" <<std::endl;
    if (weight.type == nvinfer1::DataType::kFLOAT)
    {
        _h_weight.assign((float*)weight.values,(float*)weight.values+weight.count);
    } else { throw std::runtime_error("Unsupported  weight dtype");}

    if (bias.type == nvinfer1::DataType::kFLOAT)
    {
        _h_bias.assign((float*)bias.values,(float*)bias.values+bias.count);
    } else { throw std::runtime_error("Unsupported  bias dtype");}
    cublasCreate(&_cublas_handle);
}


DCNv2Plugin::DCNv2Plugin(int in_channel,
                         int out_channel,
                         int kernel_H,
                         int kernel_W,
                         int deformable_group,
                         int dilation,
                         int groups,
                         int padding,
                         int stride,
                         const std::vector<float> &weight, const std::vector<float> &bias):_in_channel(in_channel),
                        _out_channel(out_channel),_kernel_H(kernel_H),_kernel_W(kernel_W),_deformable_group(deformable_group),
                         _dilation(dilation),_groups(groups),_padding(padding),_stride(stride),_initialized(false){

    //std::cout << "**************************** call DCNv2 construct ******************************" <<std::endl;

    _h_weight.assign(weight.begin(), weight.end());
    _h_bias.assign(bias.begin(), bias.end());
    cublasCreate(&_cublas_handle);
}

int DCNv2Plugin::initialize() {
    //std::cout << "**************************** call DCNv2 initialize ******************************" <<std::endl;
    if(_initialized) return 0;
    auto output_h = (_input_dims.d[2] + 2 * _padding - (_dilation * (_kernel_H - 1) + 1)) / _stride + 1;
    auto output_w = (_input_dims.d[3] + 2 * _padding - (_dilation * (_kernel_H - 1) + 1)) / _stride + 1;
    size_t ones_size = output_h * output_w * sizeof(float);
    size_t weight_size = _h_weight.size()* sizeof(float);
    size_t bias_size = _h_bias.size()* sizeof(float);
    float *ones_cpu = new float[ones_size/ sizeof(float)];
    for (int i = 0; i < ones_size/ sizeof(float); i++) {
        ones_cpu[i] = 1.0;
    }
    CHECK_CUDA(cudaMalloc((void**)&_d_columns, _in_channel * _kernel_H * _kernel_W * ones_size););
    CHECK_CUDA(cudaMalloc((void**)&_d_ones, ones_size));
    CHECK_CUDA(cudaMalloc((void**)&_d_weight, weight_size));
    CHECK_CUDA(cudaMalloc((void**)&_d_bias, bias_size));
    CHECK_CUDA(cudaMemcpy(_d_ones, ones_cpu, ones_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_d_weight, _h_weight.data(), weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_d_bias, _h_bias.data(), bias_size, cudaMemcpyHostToDevice));
    delete[] ones_cpu;
    _initialized = true;

    return 0;
}
void DCNv2Plugin::terminate() {
    //std::cout << "**************************** call DCNv2 terminate ******************************" <<std::endl;
    if (!_initialized) {
        return;
    }
    cudaFree(_d_columns);
    cudaFree(_d_bias);
    cudaFree(_d_weight);
    cudaFree(_d_ones);
    cublasDestroy(_cublas_handle);
    _initialized = false;
}

DCNv2Plugin::~DCNv2Plugin() {
    //std::cout << "**************************** call DCNv2 deconstruct ******************************" <<std::endl;
    terminate();
}


void DCNv2Plugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  //std::cout << "**************************** call DCNv2 configurePlugin ******************************" <<std::endl;
  assert(nbInputs == 3);
  assert(nbOutputs == 1);
  auto &input_desc = in[0].desc;
  _input_dims = input_desc.dims;
}

bool DCNv2Plugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
  //std::cout << "**************************** call DCNv2 supportsFormatCombination ******************************" <<std::endl;
  assert(nbInputs == 3);
  assert(nbOutputs == 1);
  assert(pos < (nbInputs + nbOutputs));
  return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == nvinfer1::TensorFormat::kNCHW);
}

nvinfer1::DimsExprs DCNv2Plugin::getOutputDimensions(int outputIndex,
                                                     const nvinfer1::DimsExprs* inputs,
                                                     int nbInputs,
                                                     nvinfer1::IExprBuilder& exprBuilder) {
  assert(outputIndex == 0);
  //std::cout << "**************************** call DCNv2 getOutputDimensions ******************************" <<std::endl;
  assert(nbInputs == 3);
  nvinfer1::DimsExprs output(inputs[0]);
  auto input_h = output.d[2]->getConstantValue();
  auto input_w = output.d[3]->getConstantValue();
  std::cout << _padding <<std::endl;
  std::cout << _dilation <<std::endl;
  std::cout << _kernel_H<<std::endl;
  std::cout << _kernel_W<<std::endl;
  std::cout << _stride <<std::endl;
  auto output_h = (input_h + 2 * _padding - (_dilation * (_kernel_H - 1) + 1)) / _stride + 1;
  auto output_w = (input_w + 2 * _padding - (_dilation * (_kernel_W - 1) + 1)) / _stride + 1;
  std::cout << output_h <<std::endl;
  std::cout << output_w <<std::endl;
  std::cout << _out_channel <<std::endl;
  output.d[1] = exprBuilder.constant(_out_channel);
  output.d[2] = exprBuilder.constant(output_h);
  output.d[3] = exprBuilder.constant(output_w);
  return output;
}

size_t DCNv2Plugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

int DCNv2Plugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                         const nvinfer1::PluginTensorDesc* outputDesc,
                         const void* const* inputs, void* const* outputs,
                         void* workspace,  cudaStream_t stream) {
    std::cout << "**************************** call DCNv2 enqueue ******************************" <<std::endl;
    std::cout << "batch dim: " << inputDesc[0].dims.d[0] << std::endl;
    float alpha ,beta;
    int m, n, k;

    const float* input = static_cast<const float *>(inputs[0]);
    const float* offset = static_cast<const float *>(inputs[1]);
    const float* mask = static_cast<const float *>(inputs[2]);
    float * output = static_cast<float *>(outputs[0]);
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int b = input_dims.d[0];
    assert(b == 1);
    int h = input_dims.d[2];
    int w = input_dims.d[3];
    assert(h == _input_dims.d[2]);
    assert(w == _input_dims.d[3]);

    int height_out = (h + 2 * _padding - (_dilation * (_kernel_H - 1) + 1)) / _stride + 1;
    int width_out = (w + 2 * _padding - (_dilation * (_kernel_W - 1) + 1)) / _stride + 1;
    m = _out_channel;
    n = height_out * width_out;
    k = 1;
    alpha = 1.0;
    beta = 0.0;
    /// output  nxm
    /// ones    1xn  T ->> nx1
    /// bias    1xm
    /// ones x bias = nxm
    //  add bias
    cublasSgemm(_cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,&alpha,
                _d_ones, k,
                _d_bias, k,&beta,
                output, n);
    // im2col (offset and mask)
    modulated_deformable_im2col_cuda(stream,input,offset,mask,
                                     1, _in_channel, h, w,
                                     height_out, width_out, _kernel_H, _kernel_W,
                                     _padding, _padding, _stride, _stride, _dilation, _dilation,
                                     _deformable_group, _d_columns);
    m = _out_channel;
    n = height_out * width_out;
    k = _in_channel * _kernel_H * _kernel_W;
    alpha = 1.0;
    beta = 1.0;
    // im2col conv
    cublasSgemm(_cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,&alpha,
                _d_columns, n,
                _d_weight, k,
                &beta,
                output, n);
    return 0;
}

void DCNv2Plugin::destroy() {
    //std::cout << "**************************** call DCNv2 destroy******************************" <<std::endl;
  delete this;
}


nvinfer1::DataType DCNv2Plugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  assert(index == 0);
  assert(nbInputs == 3);
  assert(inputTypes[0] == nvinfer1::DataType::kFLOAT);
  return inputTypes[0];
}


nvinfer1::IPluginV2DynamicExt* DCNv2Plugin::clone() const {
  IPluginV2DynamicExt* plugin = new DCNv2Plugin(_in_channel, _out_channel,
                                                _kernel_H, _kernel_W, _deformable_group,
                                                _dilation, _groups, _padding, _stride,
                                                _h_weight, _h_bias);
  plugin->setPluginNamespace(_plugin_namespace);
  return plugin;
}

PluginFieldCollection DCNv2PluginCreator::mFC{};
std::vector<PluginField> DCNv2PluginCreator::mPluginAttributes;


DCNv2PluginCreator::DCNv2PluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("in_channel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("out_channel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("kernel_h", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("kernel_w", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("deformable_group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("groups", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weight", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DCNv2PluginCreator::getPluginName() const
{
    return "DCNv2";
}

const char* DCNv2PluginCreator::getPluginVersion() const
{
    return "001";
}

const PluginFieldCollection* DCNv2PluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* DCNv2PluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    std::vector<float> weight;
    std::vector<float> bias;
    int in_channel, out_channel, kernel_h, kernel_w, deformable_group, groups, padding, stride, dilation;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "in_channel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            in_channel = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "out_channel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            out_channel = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "kernel_h"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            kernel_h = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "kernel_w"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            kernel_w = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "deformable_group"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            deformable_group = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "dilation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            dilation = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            stride = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            padding = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "groups"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            groups = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "weight"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            weight.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                weight.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "bias"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            bias.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                bias.push_back(*w);
                w++;
            }
        }
    }

    Weights weight_w{DataType::kFLOAT, weight.data(), (int64_t) weight.size()};
    Weights bias_w{DataType::kFLOAT, bias.data(), (int64_t) bias.size()};

    DCNv2Plugin* obj = new DCNv2Plugin(in_channel, out_channel, kernel_h, kernel_w, deformable_group,
                                       dilation, groups, padding, stride, weight_w, bias_w);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* DCNv2PluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    DCNv2Plugin* obj = new DCNv2Plugin{serialData, serialLength}; 
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
