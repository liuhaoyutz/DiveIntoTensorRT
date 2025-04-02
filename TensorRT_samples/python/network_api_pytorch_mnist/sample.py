#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys

# This sample uses an MNIST PyTorch model to create a TensorRT Inference Engine
import model
import numpy as np

import tensorrt as trt

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(
        name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE
    )  # 为Network指定input

    """
    add_matmul_as_fc 函数实现了在TensorRT网络中添加一个矩阵乘法操作，该操作模拟了一个全连接层（Fully Connected Layer, FC）。
    它接受输入张量，并使用提供的权重和偏置来执行线性变换。
    """
    def add_matmul_as_fc(net, input, outputs, w, b):
        assert len(input.shape) >= 3

        # 如果输入张量恰好有3个维度，则假定批次大小为1；否则，从输入张量的第一个维度获取批次大小m。
        m = 1 if len(input.shape) == 3 else input.shape[0]
        
        # 计算每个样本的特征数量k，即除了批次大小外的所有元素的数量。
        k = int(np.prod(input.shape) / m)
        assert np.prod(input.shape) == m * k

        """
        计算输出特征的数量（即全连接层的输出维度）。
        w.size是权重矩阵w中所有元素的总数。
        k表示输入特征的数量，即每个样本的特征维度大小。
        w.size / k这个除法操作假设权重矩阵w的形状是[n, k]，其中n是输出特征的数量，而k是输入特征的数量。
        因此，通过将权重矩阵中的总元素数量w.size除以输入特征数量k，我们可以得到输出特征的数量n。
        """
        n = int(w.size / k)
        assert w.size == n * k
        assert b.size == n

        # 调用INetworkDefinition接口的add_shuffle函数，将一个shuffle层添加到Network。
        # 注意返回值是新添加的shuffle层。shuffle层的作用包括2次transpose和一次reshape。
        input_reshape = net.add_shuffle(input)

        # 设置shuffle层（即IShuffleLayer类）的reshape_dims属性为trt.Dims2(m, k)，
        # 这个属性代表the reshaped dimensions. shuffle layer将input转换为维度为(m, k)。
        input_reshape.reshape_dims = trt.Dims2(m, k)

        # 调用INetworkDefinition接口的add_constant函数，向Network中添加一个constant layer。即维度为(n, k)的w。
        filter_const = net.add_constant(trt.Dims2(n, k), w)

        # 调用INetworkDefinition接口的add_matrix_multiply函数。
        # 将input_reshape和转置后的filter_const做矩阵乘法。得到的输出矩阵维度为(m, n)。
        mm = net.add_matrix_multiply(
            input_reshape.get_output(0),
            trt.MatrixOperation.NONE,
            filter_const.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )

        # 调用INetworkDefinition接口的add_constant函数，向Network中添加一个constant layer。即维度为(1, n)的b。
        bias_const = net.add_constant(trt.Dims2(1, n), b)

        # 调用INetworkDefinition接口的add_elementwise函数。
        # 将mm和bias_const做逐元素加法。
        bias_add = net.add_elementwise(
            mm.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM
        )

        # 调用INetworkDefinition接口的add_shuffle函数，将一个shuffle层添加到Network。
        output_reshape = net.add_shuffle(bias_add.get_output(0))
        # 设置shuffle层（即IShuffleLayer类）的reshape_dims属性为trt.Dims4(m, n, 1, 1)。
        # 这样，通过shuffle层处理，得到的output_reshape的维度为(m, n, 1, 1)。
        output_reshape.reshape_dims = trt.Dims4(m, n, 1, 1)

        return output_reshape

    conv1_w = weights["conv1.weight"].cpu().numpy()  # 从weights中取出conv1的权重
    conv1_b = weights["conv1.bias"].cpu().numpy()  # 从weights中取出conv1的偏置
    conv1 = network.add_convolution_nd(  # 将一个卷积层添加到Network，注意add_convolution_nd方法的返回值是新添加到Network中的卷积层
        input=input_tensor,
        num_output_maps=20,
        kernel_shape=(5, 5),
        kernel=conv1_w,
        bias=conv1_b,
    )
    conv1.stride_nd = (1, 1)  # 设置卷积层（即IConvolutionLayer类）的stride_nd属性。

    pool1 = network.add_pooling_nd(  # 在卷积层后面添加一个池化层，注意，返回值是新添加到Network中的池化层
        input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2)
    )
    pool1.stride_nd = trt.Dims2(2, 2)  # 设置池化层（即IPoolingLayer类）的stride_nd属性

    conv2_w = weights["conv2.weight"].cpu().numpy()  # 从weights读取conv2的权重
    conv2_b = weights["conv2.bias"].cpu().numpy()  # 从weights读取conv2的偏置
    conv2 = network.add_convolution_nd(  # 将第二个卷积层添加到池化层后面
        pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b
    )
    conv2.stride_nd = (1, 1)  # 设置第二个卷积层的stride_nd属性

    pool2 = network.add_pooling_nd(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))  # 在第二个卷积层后面添加第二个池化层
    pool2.stride_nd = trt.Dims2(2, 2)  # 设置第二个池化层的stride_nd属性

    fc1_w = weights["fc1.weight"].cpu().numpy()  # 从weights读取fc1的权重
    fc1_b = weights["fc1.bias"].cpu().numpy()  # 从weights读取fc1的偏置
    fc1 = add_matmul_as_fc(network, pool2.get_output(0), 500, fc1_w, fc1_b)  # 将fc1层添加到第二个池化层后面

    # 调用INetworkDefinition接口的add_activation函数，添加一个激活函数层，这里指定激活函数为RELU。
    relu1 = network.add_activation(
        input=fc1.get_output(0), type=trt.ActivationType.RELU
    )

    # 从weights中读取fc2的权重和偏置。
    fc2_w = weights["fc2.weight"].cpu().numpy()
    fc2_b = weights["fc2.bias"].cpu().numpy()

    # 调用add_matmul_as_fc函数，添加fc2层。
    fc2 = add_matmul_as_fc(
        network, relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b
    )

    # 将fc2层的输出的name属性设置为ModelData.OUTPUT_NAME。
    fc2.get_output(0).name = ModelData.OUTPUT_NAME

    # 调用INetworkDefinition接口的mark_output函数，mark tensor as an output。
    network.mark_output(tensor=fc2.get_output(0))


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    builder = trt.Builder(TRT_LOGGER)  # 创建builder
    network = builder.create_network(0)  # 创建一个空的network
    config = builder.create_builder_config()  # 创建config
    runtime = trt.Runtime(TRT_LOGGER)  # 创建runtime

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))  # 设置config中的memory pool limit
    # Populate the network using weights from the PyTorch model.
    populate_network(network, weights)  # 将PyTorch model转换为TensorRT Netwrok
    # Build and return an engine.
    plan = builder.build_serialized_network(network, config)  # 创建序列化的engine，这种序列化的engine在TensorRT中被称为plan
    return runtime.deserialize_cuda_engine(plan)  # 将plan转换为非序列化格式(即engine)并返回


# Loads a random test case from pytorch's DataLoader
def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = model.get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output


def main():
    common.add_help(description="Runs an MNIST network using a PyTorch model")
    # Train the PyTorch model
    mnist_model = model.MnistModel()  # 创建定义在model.py文件中的MnistModel类对象
    mnist_model.learn()  # 调用MnistModel的learn方法基于MNIST数据集训练模型
    weights = mnist_model.get_weights()  # 调用MnistModel的get_weights方法，取得模型权重
    # Do inference with TensorRT.
    engine = build_engine(weights)  # 以weights为参数，调用build_engine创建engine

    # Build an engine, allocate buffers and create a stream.
    # For more information on buffer allocation, refer to the introductory samples.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()  # 创建execution context

    # 随机选一张测试图片放到inputs[0].host指向的buffer中，其代表的数字返回给case_num变量。
    case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
    # For more information on performing inference, refer to the introductory samples.
    # The common.do_inference function will return a list of outputs - we only have one in this case.
    [output] = common.do_inference(  # 执行推理
        context,
        engine=engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    pred = np.argmax(output)  # 得到推理结果。
    common.free_buffers(inputs, outputs, stream)  # 释放分配的内存。
    print("Test Case: " + str(case_num))
    print("Prediction: " + str(pred))


if __name__ == "__main__":
    main()
