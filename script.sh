#!/bin/bash


N="1280 2560 3840 5120 6400 7680 8960 10240 11520 12800 14080 15360 16640 17920 19200 20480 21760 23040 24320 25600 26880 28160 29440 30720"
BS="64 128 256 320 640 1280"
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

for n in $N
do
  cat $prefix/$n/*/*.csv > ${n}.csv
  sort -u -n ${n}.csv > ${n}_all.csv
  ./analysis.r ${n}_all.csv
done

rm -f captureTime.csv
rm -f instantiationTime.csv
rm -f launchingTime.csv
