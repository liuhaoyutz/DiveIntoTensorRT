/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleNonZeroPlugin.cpp
//! This file contains a sample demonstrating a plugin for NonZero.
//! It can be run with the following command line:
//! Command: ./sample_non_zero_plugin [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "nonZeroKernel.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

std::string const kSAMPLE_NAME = "TensorRT.sample_non_zero_plugin";

using half = __half;

/*
这是一个辅助函数，用于处理非零元素的索引计算，这个函数根据输入的数据类型（kFLOAT / kHALF）给模板函数nonZeroIndicesImpl传递不同参数类型（float / half）来执行实际的工作。
它被NonZeroPlugin::enqueue函数调用。
这个函数接受多个参数，包括数据类型、输入数组、输出索引数组、非零元素计数、可能的阈值K、维度R和C（行数和列数）、行优先标志以及CUDA流对象。

nonZeroIndicesImpl函数的实现在TensorRT_samples/sampleNonZeroPlugin/nonZeroKernel.cu文件中。
*/
void nonZeroIndicesHelper(nvinfer1::DataType type, void const* X, void* indices, void* count, void const* K, int32_t R,
    int32_t C, bool rowOrder, cudaStream_t stream)
{
    if (type == nvinfer1::DataType::kFLOAT)
    {
        nonZeroIndicesImpl<float>(static_cast<float const*>(X), static_cast<int32_t*>(indices),
            static_cast<int32_t*>(count), static_cast<int32_t const*>(K), R, C, rowOrder, stream);
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        nonZeroIndicesImpl<half>(static_cast<half const*>(X), static_cast<int32_t*>(indices),
            static_cast<int32_t*>(count), static_cast<int32_t const*>(K), R, C, rowOrder, stream);
    }
    else
    {
        ASSERT(false && "Unsupported data type");
    }
}

/*
定义NonZeroPlugin类。该插件用于计算输入张量中非零元素的索引，并返回这些索引以及一个表示非零元素数量的张量。
*/

/*
NonZeroPlugin类实现了TensorRT中的多个接口（IPluginV3, IPluginV3OneCore, IPluginV3OneBuild, 和 IPluginV3OneRuntime）。

IPluginV3:
Plugin class for the V3 generation of user-implemented layers.
IPluginV3 acts as a wrapper around the plugin capability interfaces that define the actual behavior of the plugin.

IPluginV3OneCore:
A plugin capability interface that enables the core capability (PluginCapabilityType::kCORE).

IPluginV3OneBuild:
A plugin capability interface that enables the build capability (PluginCapabilityType::kBUILD). Exposes methods that allow the expression of the build time properties and behavior of a plugin.

IPluginV3OneRuntime:
A plugin capability interface that enables the runtime capability (PluginCapabilityType::kRUNTIME). Exposes methods that allow the expression of the runtime properties and behavior of a plugin.
*/
class NonZeroPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    NonZeroPlugin(NonZeroPlugin const& p) = default;

    NonZeroPlugin(bool rowOrder)
        : mRowOrder(rowOrder)  // 用rowOrder初始化成员变量mRowOrder，用于指示输出索引是否应该以行优先顺序排列。
    {
        initFieldsToSerialize();
    }

    /*
    负责初始化那些需要被序列化的数据成员。这个方法确保了在创建或克隆插件时，所有必要的成员变量都被正确地准备好了，以便它们可以被保存（序列化）和恢复（反序列化）。
    */
    void initFieldsToSerialize()
    {
        /*
        注意，mDataToSerialize成员在NonZeroPlugin类最后定义，它是一个vector，其定义如下：
        std::vector<nvinfer1::PluginField> mDataToSerialize;
        */

        // 清空mDataToSerialize容器中的任何先前内容。
        mDataToSerialize.clear();

        /*
        使用了emplace_back方法直接在容器的末尾构造一个新的PluginField对象，避免了额外的复制操作。
        "rowOrder"是字段的名字，用来标识这个特定的参数。
        &mRowOrder是指向实际成员变量mRowOrder的指针，表示要将它序列化。
        PluginFieldType::kINT32 指定了这个字段的数据类型是32位整数。这里将布尔值转换为了整数，因为TensorRT期望所有的序列化字段都是整数、浮点数等基本类型。
        1 表示这个字段包含一个元素。
        */
        mDataToSerialize.emplace_back(PluginField("rowOrder", &mRowOrder, PluginFieldType::kINT32, 1));

        // 更新mFCToSerialize中的nbFields成员，它记录了有多少个字段需要被序列化。mDataToSerialize.size()返回容器中元素的数量。
        mFCToSerialize.nbFields = mDataToSerialize.size();

        // 设置 mFCToSerialize的fields成员为指向mDataToSerialize内部数据的指针。
        mFCToSerialize.fields = mDataToSerialize.data();
    }

    // IPluginV3 methods

    /*
    这个方法根据传入的能力类型（PluginCapabilityType），返回相应的接口指针，以支持不同的操作阶段，如构建、运行时和核心功能。
    noexcept：该关键字表示此函数不会抛出异常。这是对调用者的承诺，意味着如果函数内部发生任何异常，它将通过其他方式处理，而不是让异常传播到调用者。
    override：表明此方法覆盖了基类中的同名虚函数。它确保编译器检查是否确实有一个匹配的虚函数在基类中声明，从而避免拼写错误或签名不匹配的问题。
    try-catch 块：尽管函数声明为 noexcept，但它还是包含了一个 try-catch 块来捕获可能发生的异常，并将错误信息记录下来。这种方式可以防止异常逃逸出函数，同时保证程序能够继续执行下去。
    */
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        try
        {
            // 如果请求的是构建阶段的能力，则将当前对象转换为IPluginV3OneBuild*类型并返回。这使得可以访问与构建相关的接口方法。
            if (type == PluginCapabilityType::kBUILD)
            {
                return static_cast<IPluginV3OneBuild*>(this);
            }
            // 如果请求的是运行时阶段的能力，则将当前对象转换为IPluginV3OneRuntime*类型并返回。这使得可以访问与运行时相关的接口方法。
            if (type == PluginCapabilityType::kRUNTIME)
            {
                return static_cast<IPluginV3OneRuntime*>(this);
            }
            // 如果能力类型是核心能力，则将当前对象转换为 IPluginV3OneCore* 类型并返回。这使得可以访问核心接口方法。
            ASSERT(type == PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore*>(this);
        }
        catch (std::exception const& e)
        {
            sample::gLogError << e.what() << std::endl;
        }
        return nullptr;
    }

    // 实现了IPluginV3类的纯虚函数clone。
    IPluginV3* clone() noexcept override
    {
        auto clone = std::make_unique<NonZeroPlugin>(*this);
        clone->initFieldsToSerialize();
        return clone.release();
    }

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override
    {
        return "NonZeroPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "0";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override
    {
        return 2;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    /*
    supportsFormatCombination 方法用于验证给定的输入和输出张量格式组合是否被支持。
    这个方法在构建阶段由TensorRT调用，以确保插件能够正确处理指定的数据类型和格式。
    */
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        bool typeOk{false};
        if (pos == 0)  // 对于第一个位置（通常为输入），检查其数据类型是否为DataType::kFLOAT或DataType::kHALF（即单精度浮点数或半精度浮点数）。
        {
            typeOk = inOut[0].desc.type == DataType::kFLOAT || inOut[0].desc.type == DataType::kHALF;
        }
        else if (pos == 1)  // 对于第二个位置（可能是第一个输出），检查其数据类型是否为 DataType::kINT32。
        {
            typeOk = inOut[1].desc.type == DataType::kINT32;
        }
        else // pos == 2  // 对于第三个位置（可能是第二个输出），同样检查其数据类型是否为DataType::kINT32。
        {
            // size tensor outputs must be NCHW INT32
            typeOk = inOut[2].desc.type == DataType::kINT32;
        }

        /*
        确保所有张量的格式为线性（PluginFormat::kLINEAR）。这意味着张量是以连续的内存块存储的，而不是其他更复杂的布局（如通道优先等）。
        最终返回值是格式检查和类型检查的结果的逻辑与。只有当两者都满足时，才认为当前格式组合被支持。
        */
        return inOut[pos].desc.format == PluginFormat::kLINEAR && typeOk;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = DataType::kINT32;
        outputTypes[1] = DataType::kINT32;
        return 0;
    }

    /*
    getOutputShapes方法的设计是为了动态推断NonZeroPlugin的输出张量尺寸。它基于输入张量的尺寸和其他配置参数（如mRowOrder），
    并通过表达式生成器创建适当的尺寸表达式。这种方法允许插件在不同的输入条件下自适应地调整其输出尺寸，从而提高灵活性和通用性。
    */
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        // The input tensor must be 2-D
        if (inputs[0].nbDims != 2)  // 如果第一个输入张量不是2维，则返回-1表示错误。这确保了插件只处理符合预期格式的输入。
        {
            return -1;
        }

        outputs[0].nbDims = 2;  // 设置第一个输出张量为2维。

        // 计算输入张量所有元素的总数（即行数乘以列数），并将结果存储在upperBound中。
        auto upperBound = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *inputs[0].d[1]);

        // 估计非零元素数量：假设平均有一半的元素是非零的，将upperBound除以2，并向下取整，得到optValue。
        // On average, we can assume that half of all elements will be non-zero
        auto optValue = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));

        // 创建一个大小张量numNonZeroSizeTensor，其值介于optValue和upperBound之间。
        auto numNonZeroSizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);

        /*
        设置输出张量的具体尺寸。根据mRowOrder决定输出顺序：
        如果mRowOrder为false，则第一个维度固定为2（表示坐标），第二个维度为numNonZeroSizeTensor（表示非零元素的数量）。
        如果mRowOrder为true，则第一个维度为numNonZeroSizeTensor，第二个维度固定为2。
        */
        if (!mRowOrder)
        {
            outputs[0].d[0] = exprBuilder.constant(2);
            outputs[0].d[1] = numNonZeroSizeTensor;
        }
        else
        {
            outputs[0].d[0] = numNonZeroSizeTensor;
            outputs[0].d[1] = exprBuilder.constant(2);
        }

        // 设置第二个输出张量为大小张量，大小张量必须声明为0维，因为它们实际上表示的是标量值。
        // output at index 1 is a size tensor
        outputs[1].nbDims = 0; // size tensors must be declared as 0-D

        return 0;
    }

    // IPluginV3OneRuntime methods

    /*
    enqueue方法的设计是为了高效地处理非零元素的索引提取。它通过CUDA内核来并行处理输入张量中的每个元素，并根据mRowOrder参数决定输出格式。

    param inputDesc, how to interpret the memory for the input tensors.
    param outputDesc, how to interpret the memory for the output tensors.
    param inputs, the memory for the input tensors.
    param outputs, the memory for the output tensors.
    param workspace, workspace for execution.
    param stream, the stream in which to execute the kernels.
    */
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        // 获取输入张量维度，R和C分别代表输入张量的行数和列数。
        int32_t const R = inputDesc[0].dims.d[0];
        int32_t const C = inputDesc[0].dims.d[1];

        auto type = inputDesc[0].type;
        // 如果输入张量的数据类型不是nvinfer1::DataType::kHALF或nvinfer1::DataType::kFLOAT，则记录错误并返回-1。
        if (!(type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT))
        {
            sample::gLogError << "Unsupported: Sample only supports DataType::kHALF and DataType::FLOAT" << std::endl;
            return -1;
        }

        // 将第二个输出张量（即大小张量）初始化为0。
        cudaMemsetAsync(outputs[1], 0, sizeof(int32_t), stream);

        // 如果workspace为空，则记录错误并返回-1。
        if (workspace == nullptr)
        {
            sample::gLogError << "Unsupported: workspace is null" << std::endl;
            return -1;
        }

        /*
        如果 mRowOrder为 false（即列优先顺序）：
        初始化工作空间：使用cudaMemsetAsync将工作空间初始化为0。
        第一次调用nonZeroIndicesHelper：只计算总的非零元素数量，并将其存储在workspace中。
        第二次调用nonZeroIndicesHelper：实际将非零索引写入outputs[0]缓冲区，并更新大小张量outputs[1]。

        如果mRowOrder为true（即行优先顺序），则直接调用nonZeroIndicesHelper一次，同时处理非零索引和大小张量。
        */
        if (!mRowOrder)
        {
            // When constructing a column major output, the kernel needs to be aware of the total number of non-zero
            // elements so as to write the non-zero indices at the correct places. Therefore, we will launch the kernel
            // twice: first, only to calculate the total non-zero count, which will be stored in workspace; and
            // then to actually write the non-zero indices to the outputs[0] buffer.
            cudaMemsetAsync(workspace, 0, sizeof(int32_t), stream);
            nonZeroIndicesHelper(type, inputs[0], nullptr, workspace, 0, R, C, mRowOrder, stream);
            nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], workspace, R, C, mRowOrder, stream);
        }
        else
        {
            nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], 0, R, C, mRowOrder, stream);
        }

        return 0;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFCToSerialize;
    }

    /*
    getWorkspaceSize方法告知TensorRT插件需要多少临时内存来完成其任务。
    在这个特定的例子中，插件只需要一个小的工作空间来存储一个32位整数，这可能是为了保存某些中间结果或状态信息。
    */
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        return sizeof(int32_t);
    }

private:
    bool mRowOrder{true};  // 输出索引是否应该以行优先顺序排列。

    // 保存要序列化的数据。用于存储nvinfer1::PluginField对象。nvinfer1::PluginField是TensorRT中定义的一个结构体，用来描述插件的配置参数。
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    
    // mFCToSerialize是nvinfer1::PluginFieldCollection对象，用于描述有几个PluginField，以及指向PluginField的指针。
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

// NonZeroPluginCreator类实现了IPluginCreatorV3One接口。
class NonZeroPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    NonZeroPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("rowOrder", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    char const* getPluginName() const noexcept override
    {
        return "NonZeroPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "0";
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFC;
    }

    /*
    遍历传入的PluginFieldCollection，查找名为"rowOrder"的字段，并将其值转换为布尔值。
    使用解析后的rowOrder参数创建一个新的NonZeroPlugin实例并返回。
    */
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override
    {
        try
        {
            bool rowOrder{true};
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                auto const fieldName(fc->fields[i].name);
                if (std::strcmp(fieldName, "rowOrder") == 0)
                {
                    rowOrder = *static_cast<bool const*>(fc->fields[i].data);
                }
            }
            return new NonZeroPlugin(rowOrder);
        }
        catch (std::exception const& e)
        {
            sample::gLogError << e.what() << std::endl;
        }
        return nullptr;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

namespace
{
struct NonZeroParams : public samplesCommon::SampleParams
{
    bool rowOrder{true};
};
} // namespace


/*
SampleNonZeroPlugin 类的设计目的是为了提供一个完整的流程来创建、配置和运行包含 NonZero 插件的 TensorRT 网络。
*/

//! \brief  The SampleNonZeroPlugin class implements a NonZero plugin
//!
//! \details The plugin is able to output the non-zero indices in row major or column major order
//!
class SampleNonZeroPlugin
{
public:
    SampleNonZeroPlugin(NonZeroParams const& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
        mSeed = static_cast<uint32_t>(time(nullptr));
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    NonZeroParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    uint32_t mSeed{};

    //!
    //! \brief Creates a TensorRT network and inserts a NonZero plugin
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(samplesCommon::BufferManager const& buffers);

    //!
    //! \brief Verifies the result
    //!
    bool verifyOutput(samplesCommon::BufferManager const& buffers);
};

/*
与sampleOnnxMNIST需要parse ONNX模型创建Network不同，本例的Network只有一个custom layer，
该custom layer包含plugin，后面推理时执行的也只是这一个custom layer。
*/
//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates a network containing a NonZeroPlugin and builds
//!          the engine that will be used to run the plugin (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleNonZeroPlugin::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    /*
    与非plugin代码相比，最大区别在于这里的注册NonZeroPluginCreator。
    */
    auto pluginCreator = std::make_unique<NonZeroPluginCreator>();
    getPluginRegistry()->registerCreator(*pluginCreator.get(), "");

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 2);

    ASSERT(network->getNbOutputs() == 2);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Creates a network with a single custom layer containing the NonZero plugin and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the NonZero plugin
//!
//! \param builder Pointer to the engine builder
//!
bool SampleNonZeroPlugin::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    /*
    使用随机数生成器和均匀分布来动态生成输入张量的行数(R)和列数(C)，范围为[10, 25]。
    */
    std::default_random_engine generator(mSeed);
    std::uniform_int_distribution<int32_t> distr(10, 25);
    int32_t const R = distr(generator);
    int32_t const C = distr(generator);

    /*
    创建一个名为"Input"的输入张量，数据类型为DataType::kFLOAT，维度为{2, {R, C}}（即 2D 张量）。
    将Input做为Network的输入张量。在后面分析的processInput方法中，默认会随机读取minist数据集（.pgm文件）的内容作为输入。
    */
    auto* in = network->addInput("Input", DataType::kFLOAT, {2, {R, C}});
    ASSERT(in != nullptr);

    std::vector<PluginField> const vecPF{{"rowOrder", &mParams.rowOrder, PluginFieldType::kINT32, 1}};
    PluginFieldCollection pfc{static_cast<int32_t>(vecPF.size()), vecPF.data()};

    auto pluginCreator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("NonZeroPlugin", "0", ""));
    // 调用NonZeroPluginCreator的createPlugin方法创建plugin。
    auto plugin = std::unique_ptr<IPluginV3>(pluginCreator->createPlugin("NonZeroPlugin", &pfc, TensorRTPhase::kBUILD));

    std::vector<ITensor*> inputsVec{in};
    // 创建包含plugin的layer，加入到Network中。
    auto pluginNonZeroLayer = network->addPluginV3(inputsVec.data(), inputsVec.size(), nullptr, 0, *plugin);
    ASSERT(pluginNonZeroLayer != nullptr);
    ASSERT(pluginNonZeroLayer->getOutput(0) != nullptr);
    ASSERT(pluginNonZeroLayer->getOutput(1) != nullptr);

    pluginNonZeroLayer->getOutput(0)->setName("Output0");
    pluginNonZeroLayer->getOutput(1)->setName("Output1");

    network->markOutput(*(pluginNonZeroLayer->getOutput(0)));
    network->markOutput(*(pluginNonZeroLayer->getOutput(1)));

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleNonZeroPlugin::infer()
{
    /*
    为输入张量和两个输出张量分配足够的空间。这里假设第一个输出张量的尺寸是输入张量的两倍，而第三个张量（可能用于其他用途）的尺寸为1。
    */
    // Since the data dependent output size cannot be inferred from the engine denote a sufficient size for the
    // corresponding output buffer (along with the rest of the I/O tensors)
    std::vector<int64_t> ioVolumes = {mInputDims.d[0] * mInputDims.d[1], mInputDims.d[0] * mInputDims.d[1] * 2, 1};

    /*
    创建缓冲区管理器：使用 BufferManager 管理所有 I/O 张量的主机和设备缓冲区。
    BufferManager 是一个 RAII 类型的对象，确保资源在超出作用域时自动释放。
    RAII（Resource Acquisition Is Initialization，资源获取即初始化）是一种编程惯用法，特别在C++中被广泛使用。它的核心思想是将资源的生命周期与对象的生命周期绑定在一起：资源在对象创建时分配，在对象销毁时自动释放。这种方式可以有效防止资源泄漏，并简化代码，因为开发者不需要显式地管理资源的分配和释放。
    */
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, ioVolumes);

    /*
    从mEngine创建一个新的IExecutionContext实例，并将其封装在智能指针中。
    */
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    /*
    遍历所有I/O张量，获取每个张量的名称。使用context->setTensorAddress 将每个张量的设备缓冲区地址设置到执行上下文中。
    */
    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))  // 调用processInput方法将输入数据填充到管理缓冲区中。
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));  // 创建一个 CUDA 流用于异步操作。

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);  // 将主机输入缓冲区的数据异步复制到设备输入缓冲区。

    // 调用context->enqueueV3执行推理任务，并传递CUDA流以支持异步执行。
    // 这里就会调用NonZeroPlugin类的enqueue方法，进而调用nonZeroIndicesHelper，进而调用nonZeroIndicesImpl，进而调用核函数findNonZeroIndicesKernel
    bool status = context->enqueueV3(stream);
    if (!status)
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);  // 将设备输出缓冲区的数据异步复制回主机输出缓冲区。

    // Wait for the work in the stream to complete.
    CHECK(cudaStreamSynchronize(stream));  // 等待CUDA流中的所有操作完成。

    // Release stream.
    CHECK(cudaStreamDestroy(stream));  // 释放CUDA流资源。

    // Verify results
    if (!verifyOutput(buffers))  // 调用verifyOutput方法验证推理结果。
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleNonZeroPlugin::processInput(samplesCommon::BufferManager const& buffers)
{
    // 从成员变量mInputDims中提取输入张量的高度(inputH)和宽度(inputW)。
    int32_t const inputH = mInputDims.d[0];
    int32_t const inputW = mInputDims.d[1];

    // 创建一个std::vector<uint8_t>用于存储从文件读取的图像数据，大小为inputH * inputW。
    std::vector<uint8_t> fileData(inputH * inputW);

    // 使用随机数生成器和均匀分布来随机选择一个数字（范围 [0, 9]），这个数字用于确定要读取的PGM文件名。
    std::default_random_engine generator(mSeed);
    std::uniform_int_distribution<int32_t> distr(0, 9);
    auto const number = distr(generator);
    // 调用readPGMFile函数将选定的PGM文件内容读取到fileData容器中。
    readPGMFile(locateFile(std::to_string(number) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // 通过buffers.getHostBuffer获取指向输入张量主机缓冲区的指针，并将其强制转换为float*类型。
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

    /*
    遍历fileData，对每个像素值进行归一化处理（1.0 - float(fileData[i] / 255.0)），然后将结果写入主机缓冲区。
    这里采用1.0 - ...是为了反转图像颜色，具体取决于应用需求。
    */
    for (int32_t i = 0; i < inputH * inputW; ++i)
    {
        auto const raw = 1.0 - float(fileData[i] / 255.0);
        hostDataBuffer[i] = raw;
    }

    // 将处理后的输入数据以矩阵形式打印出来。
    sample::gLogInfo << "Input:" << std::endl;
    for (int32_t i = 0; i < inputH; ++i)
    {
        for (int32_t j = 0; j < inputW; ++j)
        {
            sample::gLogInfo << hostDataBuffer[i * inputW + j];
            if (j < inputW - 1)
            {
                sample::gLogInfo << ", ";
            }
        }
        sample::gLogInfo << std::endl;
    }
    sample::gLogInfo << std::endl;

    return true;
}

//!
//! \brief Verify result
//!
//! \return whether the output correctly identifies all (and only) non-zero elements
//!
bool SampleNonZeroPlugin::verifyOutput(samplesCommon::BufferManager const& buffers)
{
    // 从buffers中获取输入张量的主机缓冲区指针，并将其强制转换为float*类型。
    float* input = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // 从buffers中获取第一个输出张量（包含非零元素索引）的主机缓冲区指针，并将其强制转换为int32_t*类型。
    int32_t* output = static_cast<int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    // 从buffers中获取第二个输出张量（包含非零元素的数量），并将其解引用为int32_t。
    int32_t count = *static_cast<int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

    // 创建一个大小为mInputDims.d[0] * mInputDims.d[1]的布尔向量covered，用于跟踪哪些输入位置被输出索引覆盖。
    std::vector<bool> covered(mInputDims.d[0] * mInputDims.d[1], false);

    // 根据mParams.rowOrder参数的不同，以不同的格式打印输出数据。
    sample::gLogInfo << "Output:" << std::endl;
    if (mParams.rowOrder)
    {
        for (int32_t i = 0; i < count; ++i)
        {
            for (int32_t j = 0; j < 2; ++j)
            {
                sample::gLogInfo << output[j + 2 * i] << " ";
            }
            sample::gLogInfo << std::endl;
        }
    }
    else
    {
        for (int32_t i = 0; i < 2; ++i)
        {
            for (int32_t j = 0; j < count; ++j)
            {
                sample::gLogInfo << output[j + count * i] << " ";
            }
            sample::gLogInfo << std::endl;
        }
    }

    // 验证输出索引的有效性，根据mParams.rowOrder参数的不同，计算索引idx并设置covered[idx]为true。如果对应的输入位置为零，则返回false。
    if (!mParams.rowOrder)
    {
        for (int32_t i = 0; i < count; ++i)
        {
            auto const idx = output[i] * mInputDims.d[1] + output[i + count];
            covered[idx] = true;
            if (input[idx] == 0.F)
            {
                return false;
            }
        }
    }
    else
    {
        for (int32_t i = 0; i < count; ++i)
        {
            auto const idx = output[2 * i] * mInputDims.d[1] + output[2 * i + 1];
            covered[idx] = true;
            if (input[idx] == 0.F)
            {
                return false;
            }
        }
    }

    // 检查covered向量中的每一个位置。如果某个位置未被覆盖（即covered[i]为false），并且对应的输入位置不是零，则返回false。
    for (int32_t i = 0; i < static_cast<int32_t>(covered.size()); ++i)
    {
        if (!covered[i])
        {
            if (input[i] != 0.F)
            {
                return false;
            }
        }
    }

    return true;
}

// 初始化NonZeroParams结构体，该结构体包含用于配置和运行TensorRT推理引擎的参数。
//!
//! \brief Initializes members of the params struct using the command line args
//!
NonZeroParams initializeSampleParams(samplesCommon::Args const& args)
{
    NonZeroParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.inputTensorNames.push_back("Input");
    params.outputTensorNames.push_back("Output0");
    params.outputTensorNames.push_back("Output1");
    params.fp16 = args.runInFp16;
    params.rowOrder = args.rowOrder;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_non_zero_plugin [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
    std::cout << "--columnOrder   Run plugin in column major output mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(kSAMPLE_NAME, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleNonZeroPlugin sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for NonZero plugin" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
