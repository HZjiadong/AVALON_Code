#!/bin/bash


N="7680 15360 23040 30720 38400"
BS="64 128 192 256 320 384 512 640 768 960 1280 1536 1920 2560 3840 7680"
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