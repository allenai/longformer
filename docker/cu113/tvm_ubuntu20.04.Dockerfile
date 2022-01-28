###### DOCKERFILE #########
# with versions
# Ubuntu 20.04
# CUDA 11.3.1
# LLVM 13
# Python 3.8.10
# Pytorch 1.10.2
# Transformers 4.16.0
# TVM 0.8.0
##########################

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing
# for debugging
RUN apt-get install -y vim 

# ubuntu_install_core.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
        git make google-mock libgtest-dev cmake wget unzip libtinfo-dev libz-dev \
        libcurl4-openssl-dev libssl-dev libopenblas-dev g++ sudo \
        apt-transport-https graphviz pkg-config curl
WORKDIR /usr/src/googletest 
RUN cmake CMakeLists.txt 
RUN make 
RUN cp lib/libgmock.a /usr/lib
RUN cp lib/libgmock_main.a /usr/lib
RUN cp lib/libgtest.a /usr/lib
RUN cp lib/libgtest_main.a /usr/lib
WORKDIR /

# LLVM
RUN echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main >> /etc/apt/sources.list.d/llvm.list
RUN echo deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main >> /etc/apt/sources.list.d/llvm.list
RUN wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -

RUN apt-get update && apt-get install -y llvm-13

# Python
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install -y python3-dev python3-setuptools


# Install pip
WORKDIR /tmp 
RUN wget -q https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py
WORKDIR /

RUN pip3 install \
    attrs \
    cloudpickle \
    cython \
    decorator \
    mypy \
    numpy \
    orderedset \
    packaging \
    Pillow \
    pytest \
    pytest-profiling \
    pytest-xdist \
    requests \
    scipy \
    synr==0.5.0 \
    six \
    tornado

RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install ipython ipdb tensorboardx tensorboard fairseq==v0.9.0
RUN pip3 install --ignore-installed transformers==4.16.0

# Build TVM
RUN git clone https://github.com/apache/tvm /usr/tvm --recursive
WORKDIR /usr/tvm
RUN git checkout 7b3a22e465dd6aca4729504a19beb4bc23312755
RUN echo set\(USE_LLVM llvm-config-13\) >> config.cmake
RUN echo set\(USE_CUDA ON\) >> config.cmake
RUN echo set\(USE_CUDNN ON\) >> config.cmake
RUN echo set\(USE_BLAS openblas\) >> config.cmake
RUN echo set\(USE_MKL OFF\) >> /usr/tvm/config.cmake
RUN echo set\(USE_MKL_PATH OFF\) >> /usr/tvm/config.cmake
RUN mkdir -p build
WORKDIR /usr/tvm/build
RUN cmake ..
RUN make -j10
WORKDIR /

# Environment variables
ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/topi/python:/usr/tvm/nnvm/python/:/usr/tvm/vta/python:${PYTHONPATH}
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}