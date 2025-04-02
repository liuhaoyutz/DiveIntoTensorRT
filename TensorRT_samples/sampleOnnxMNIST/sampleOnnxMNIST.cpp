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
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST  // 本示例的类定义
{
public:
    // 构造函数只是用params初始化了mParams成员。
    // params的初始化在下面的initializeSampleParams函数中，指定onnx文件名，input/output tensor name，量化等信息
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();  // 该函数完成engine的构建，包括Creates the network, configures the builder and creates the network engine

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();  // 运行engine，执行推理

private:
    // 保存onnx模型的相关配置参数，例如onnx文件名，input/output tensor name，量化等信息
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    // 输入维度
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    // 输出维度
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    // 因为这是一个关于MNIST数据集的例子，mNumber保存图片分类对应的数字0-9
    int mNumber{0};             //!< The number to classify

    // 保存TensorRT runtime
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    // 保存TensorRT engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    // 创建network
    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache);

    // 读取并处理input
    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    // 验证推理结果是否正确
    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    // 创建infer builder
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    // 通过infer builder创建network，这里只是初始化，network还没有内容，网络内容的具体填充在constructNetwork函数解析onnx模型时完成。
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        return false;
    }

    // 通过infer builder创建config
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    // 创建onnx parser，注意调用createParser函数时传递了network参数，所以后面用parser解析ONNX模型后，自动用解析出的权重初始化network的各个layer。
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto timingCache = SampleUniquePtr<nvinfer1::ITimingCache>();

    // 调用constructNetwork函数，通过解析onnx模型文件，填充network内容。
    auto constructed = constructNetwork(builder, network, config, parser, timingCache);
    if (!constructed)
    {
        return false;
    }

    // 创建CUDA stream
    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    // 到这里已经完成了network和config的初始化。
    // 以network和config为参数调用builder->buildSerializedNetwork函数，得到序列化的engine
    // 在TensorRT中，序列化的engine也称为plan。
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    if (timingCache != nullptr && !mParams.timingCacheFile.empty())
    {
        samplesCommon::updateTimingCacheFile(
            sample::gLogger.getTRTLogger(), mParams.timingCacheFile, timingCache.get(), *builder);
    }

    // 创建runtime
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    // 调用IRuntime类的deserializeCudaEngine函数，创建非序列化的engine。
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();  // 取得输入tensor维度
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();  // 取得输出tensor维度。
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache)
{
    // 解析onnx模型文件
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    // 根据mParams配置config。
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.bf16)
    {
        config->setFlag(BuilderFlag::kBF16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }

    // 配置timingCache。
    if (mParams.timingCacheFile.size())
    {
        timingCache = samplesCommon::buildTimingCacheFromFile(
            sample::gLogger.getTRTLogger(), *config, mParams.timingCacheFile, sample::gLogError);
    }

    // 配置DLA。
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    // 创建RAII buffer manager object，确保资源在超出作用域时自动释放。
    // RAII（Resource Acquisition Is Initialization，资源获取即初始化）是一种编程惯用法，
    // 特别在C++中被广泛使用。它的核心思想是将资源的生命周期与对象的生命周期绑定在一起：资源在对象创建时分配，
    // 在对象销毁时自动释放。这种方式可以有效防止资源泄漏，并简化代码，因为开发者不需要显式地管理资源的分配和释放。
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    // 调用ICudaEngine类的createExecutionContext方法创建execution context。
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // 调用ICudaEngine类的getNbIOTensors函数，取得IO tensors的个数，循环依次将每个tensor保存到context中
    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);  // 确认mParams.inputTensorNames向量中只有一个变量，即只能有一个input tensor。

    // 调用processInput函数，其作用是Reads the input and stores the result in a managed buffer
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();  // 将input buffer拷贝到GPU上。

    // 调用IExecutionContext类的executeV2函数以同步方式执行推理。
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // 将数据从GPU拷贝到host。
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))  // 调用verifyOutput函数验证输出结果正确。
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    // 取得input tensor的维度信息。
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    // 随机读取数据集。
    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    mNumber = rand() % 10;
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // 打印输入tensor。
    // Print an ascii representation
    sample::gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    // 遍历 fileData，对每个像素值进行归一化处理（1.0 - float(fileData[i] / 255.0)），
    // 然后将结果写入主机缓冲区。这里采用 1.0 - ... 是为了反转图像颜色，具体取决于应用需求。
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0F};
    int idx{0};

    // Calculate Softmax
    float sum{0.0F};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5F)), '*')
                         << std::endl;
    }
    sample::gLogInfo << std::endl;

    return idx == mNumber && val > 0.9F;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.bf16 = args.runInBf16;
    params.timingCacheFile = args.timingCacheFile;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << "[-t or --timingCacheFile=<path to timing cache file]" << std::endl;
    std::cout << "--help             Display help information" << std::endl;
    std::cout << "--datadir          Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N     Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8             Run in Int8 mode." << std::endl;
    std::cout << "--fp16             Run in FP16 mode." << std::endl;
    std::cout << "--bf16             Run in BF16 mode." << std::endl;
    std::cout << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
              << "created." << std::endl;
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

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    // 创建SampleOnnxMNIST对象
    SampleOnnxMNIST sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    // 创建network engine
    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    // 执行推理
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
