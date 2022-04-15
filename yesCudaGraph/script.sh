#!/bin/bash

log=log_file.txt
printf "This is the Log File of date" > $log
timer_start=`date +"%Y-%m-%d %H:%M:%S"`
echo "start Time "$timer_start >> $log
prefix=`date +"%Y-%m-%d"`
N="1280 2560 3840 5120 6400 7680 8960 10240 11520 12800 14080 15360 16640 17920 19200 20480 21760 23040 24320 25600 26880 28160 29440 30720 32000 33280 34560 35840 37120 38400"
BS="64 128 256 320 640 1280 2560 3840 5120 6400 7680 8960 10240 11520 12800 14080 15360 16640 17920 19200 20480 21760 23040 24320 25600 26880 28160 29440 30720 32000 33280 34560 35840 37120 38400"
make executable
for n in $N
do
  for bs in $BS
  do
    echo "----- run GEMM $n x $n, BS=$bs" >> $log  
    CUDA_VISIBLE_DEVICES=1 numactl -m 0 -C 0 ./executable $n $bs >> $log

    #Move .csv file to directory $dirname
    dir=$prefix/$n/$bs
    mkdir -p $dir
    mv captureTime.csv $dir
    mv instantiationTime.csv $dir
    mv launchingTime.csv $dir
  done
done

for n in $N
do
  cat $prefix/$n/*/*.csv > ${n}.csv >> $log
  sort -u -n ${n}.csv > ${n}_all.csv >> $log
  ./analysis.r ${n}_all.csv >> $log
done

timer_end=`date +"%Y-%m-%d %H:%M:%S"`
echo "end Time "$timer_end >> $log

dir2=$prefix
mv log_file.txt $dir2

dir3=${prefix}_records
mkdir $dir3
mv *.csv $dir3

dir4=${prefix}_plots
mkdir $dir4
mv *.pdf $dir4

rm -f captureTime.csv
rm -f instantiationTime.csv
rm -f launchingTime.csv