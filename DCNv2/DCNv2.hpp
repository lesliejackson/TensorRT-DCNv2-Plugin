//
// Created by cao on 19-12-20.
//

#ifndef TRT_DCNV2_PLUGIN_H
#define TRT_DCNV2_PLUGIN_H

#include "plugin.h"
#include "serialize.hpp"
#include <cudnn.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda.h>

namespace nvinfer1 {
namespace plugin {

class DCNv2Plugin final : public nvinfer1::IPluginV2DynamicExt {
private:
    int _in_channel;
    int _out_channel;
    int _kernel_H;
    int _kernel_W;
    int _deformable_group;
    int _dilation;
    int _groups; // not use
    int _padding;
    int _stride;
    std::vector<float> _h_weight;
    std::vector<float> _h_bias;
    float* _d_weight;
    float* _d_bias;
    float* _d_ones;
    float *_d_columns;
    cublasHandle_t _cublas_handle;
    nvinfer1::Dims _input_dims;
    const char* _plugin_namespace;

    bool _initialized;

public:
    void deserialize(void const* serialData, size_t serialLength) {
        deserialize_value(&serialData, &serialLength, &_in_channel);
        deserialize_value(&serialData, &serialLength, &_out_channel);
        deserialize_value(&serialData, &serialLength, &_kernel_H);
        deserialize_value(&serialData, &serialLength, &_kernel_W);
        deserialize_value(&serialData, &serialLength, &_deformable_group);
        deserialize_value(&serialData, &serialLength, &_dilation);
        deserialize_value(&serialData, &serialLength, &_groups);
        deserialize_value(&serialData, &serialLength, &_padding);
        deserialize_value(&serialData, &serialLength, &_stride);
        deserialize_value(&serialData, &serialLength, &_h_weight);
        deserialize_value(&serialData, &serialLength, &_h_bias);
    }
    size_t getSerializationSize() const override {
        return (serialized_size(_in_channel) +
                serialized_size(_out_channel) +
                serialized_size(_kernel_H) +
                serialized_size(_kernel_W) +
                serialized_size(_deformable_group) +
                serialized_size(_dilation) +
                serialized_size(_groups) +
                serialized_size(_padding) +
                serialized_size(_stride) +
                serialized_size(_h_weight) +
                serialized_size(_h_bias)
               );
    }
    void serialize(void *buffer) const override {
        serialize_value(&buffer, _in_channel);
        serialize_value(&buffer, _out_channel);
        serialize_value(&buffer, _kernel_H);
        serialize_value(&buffer, _kernel_W);
        serialize_value(&buffer, _deformable_group);
        serialize_value(&buffer, _dilation);
        serialize_value(&buffer, _groups);
        serialize_value(&buffer, _padding);
        serialize_value(&buffer, _stride);
        serialize_value(&buffer, _h_weight);
        serialize_value(&buffer, _h_bias);
    }

    DCNv2Plugin(int in_channel,
                int out_channel,
                int kernel_H,
                int kernel_W,
                int deformable_group,
                int dilation,
                int groups,
                int padding,
                int stride,
                nvinfer1::Weights const& weight,
                nvinfer1::Weights const& bias);

    DCNv2Plugin(int in_channel,
                int out_channel,
                int kernel_H,
                int kernel_W,
                int deformable_group,
                int dilation,
                int groups,
                int padding,
                int stride,
                const std::vector<float> &weight,
                const std::vector<float> &bias);

    DCNv2Plugin(void const* serialData, size_t serialLength) : _initialized(false) {
        this->deserialize(serialData, serialLength);
     }

    DCNv2Plugin() = delete;

    const char* getPluginType() const override { return "DCNv2"; }

    const char* getPluginVersion() const override { return "001"; }

    void destroy() override;

    int getNbOutputs() const override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) override;
    int initialize() override;

    void terminate() override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace,  cudaStream_t stream) override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    nvinfer1::IPluginV2DynamicExt* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override {_plugin_namespace = pluginNamespace;};

    const char* getPluginNamespace() const override {return _plugin_namespace;};

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) override {};

    void detachFromContext() override {};

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, 
                         const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    ~DCNv2Plugin();
};



class DCNv2PluginCreator : public BaseCreator
{
public:
  DCNv2PluginCreator();

  ~DCNv2PluginCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const PluginFieldCollection* getFieldNames() override;

  IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mNamespace;
};

}
}
#endif //TRT_DCNV2_PLUGIN_H
