# DiveIntoTensorRT
这个Repo的内容是TensorRT官方示例程序，添加个人注释理解。  

## 配置TensorRT开发环境

步骤一：下载TensorRT GA(General Availability ) build并配置环境变量  
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz  

cd ~/Downloads  
tar -xvzf TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz  
注意TRT_LIBPATH环境变量，github网页上少写了lib，后面编译OSS时会出错。下面3个环境变量必须设置：  
export TRT_LIBPATH=/home/haoyu/Downloads/TensorRT-10.7.0.23/lib  
export TRT_LIB_DIR=/home/haoyu/Downloads/TensorRT-10.7.0.23/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/haoyu/Downloads/TensorRT-10.7.0.23/lib  

注意，下面的步骤二和步骤三做其中一个即可。  

步骤二：下载TensorRT库源码并编译安装  
git clone -b main https://github.com/nvidia/TensorRT TensorRT  
cd TensorRT  
git submodule update --init --recursive  

注意编译前必须设置环境变量 TRT_LIBPATH，指向TensorRT-10.7.0.23/lib目录。  

#指从github下载的TensorRT源码的目录, OSS是Open Source Software的缩写  
cd $TRT_OSSPATH  
mkdir -p build && cd build  
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out  
make -j$(nproc)  
make install  

步骤三：直接pip安装TensorRT库  
pip install tensorrt  

步骤四：编译示例程序  
cd DiveIntoTensorRT/TensorRT_samples/sampleOnnxMNIST  
make  
cd ../../bin  
./sample_onnx_mnist  

步骤五：Python接口示例程序  
cd DiveIntoTensorRT/TensorRT_samples/python/network_api_pytorch_mnist  
pip install -r requirements.txt  
python sample.py  
  
## Reference  
https://github.com/NVIDIA/TensorRT  
