# Location of the CUDA Toolkit
# CUDA_HOME= /grid5000/spack/opt/spack/linux-debian11-x86_64/gcc-10.2.0/cuda-11.5.1-t5svyhvs2pr7v7v5jz5dsmi6bz54dlbi

all: executable

# Compilation
executable: main_withCudaGraph.cpp Makefile
	g++ -g main_withCudaGraph.cpp -o executable -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcublas -lcurand -fopenmp -lcudart

################################################################################

# Execution
run:
	numactl -m 0 -C 0 ./executable 1024 512


clean:
	rm -f executable