#!/bin/bash


N="4096 8192 16384 32768 65536"
BS="64 128 256 512 1024 2048 4096"
make executable
for n in $N
do
  for bs in $BS
  do
    echo "----- run GEMM $n x $n, BS=$bs"
    numactl -m 0 -C 0 ./executable $n $bs

    #Move .csv file to directory $dirname
    prefix=`date +"%Y-%m-%d"`
    dir=$prefix/$n/$bs
    mkdir -p $dir
    mv captureTime.csv $dir
    mv instantiationTime.csv $dir
    mv launchingTime.csv $dir
  done
done

cat 2022-03-25/$n/*/*.csv > ${n}.csv
sort -u -n ${n}.csv > ${n}_all.csv
./analysis.r ${n}_all.csv

rm -f captureTime.csv
rm -f instantiationTime.csv
rm -f launchingTime.csv