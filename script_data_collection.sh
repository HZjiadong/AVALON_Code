#!/bin/bash


N="4096 8192 16384"
BS="64 128 256 512 1024 2048 4096"

for n in $N
do
  for bs in $BS
  do
    echo "----- run GEMM $n x $n, BS=$bs"
    g++ -g main.cpp -o executable -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcublas -lcurand -fopenmp -lcudart
    numactl -m 0 -C 0 ./executable 10240

    #Move .csv file to directory $dirname
    prefix=`date +"%Y-%m-%d"`
    dir=$prefix/$n/$bs
    mkdir -p $dir
    mv captureTime.csv $dir
    mv instantiationTime.csv $dir
    mv launchingTime.csv $dir
    rm -f executable
  done
done