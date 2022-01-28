# Instructions to compile the TVM CUDA kernel
This assumes that docker and its gpu runtime are installed. Check the following for reference:

Docker: https://docs.docker.com/install/

Docker gpu runtime: https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster

1. clone longformer

```bash
git clone https://github.com/allenai/longformer.git
cd longformer
```

2. build docker image
```bash
docker build -t my_tvm_image -f docker/cu113/tvm_ubuntu18.04 .
```

3. run docker container
```bash
docker run -it --gpus all  --mount type=bind,source="$(pwd)",target=/code my_tvm_image
```

4. inside the docker container, do the following

```bash
cd code
mv tvm tvm_runtime  # avoid collision between our small tvm runtime and the full tvm library
rm longformer/lib/*  # remove old binaries
python3 -c "from longformer.diagonaled_mm_tvm import *; DiagonaledMM._get_function('float32', 'cuda')"  #  compile new ones
ls longformer/lib/  # check the `lib` dir for the new binaries
mv tvm_runtime tvm  # don't forget to put them back
```