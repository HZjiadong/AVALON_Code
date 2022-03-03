################################################################################
# Certains code from: Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Others: Author GUO jiadong INRIA intern
################################################################################

# Location of the CUDA Toolkit
# CUDA_HOME= /grid5000/spack/opt/spack/linux-debian11-x86_64/gcc-10.2.0/cuda-11.5.1-t5svyhvs2pr7v7v5jz5dsmi6bz54dlbi

# Compilation
g++ main.cpp -o executable -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcublas -lcurand -fopenmp -lcudart

################################################################################

# Gencode arguments
numactl -m 0 -C 0 ./executable 10240

################################################################################

# Target rules