#!/bin/bash

log=log_file_noCudaGraph.txt
printf "This is the Log File of date" > $log
timer_start=`date +"%Y-%m-%d %H:%M:%S"`
echo "start Time "$timer_start >> $log
prefix=`date +"%Y-%m-%d"`
N="1280 5120 8960 12800 16640 20480 24320 28160 32000 35840"
BS="64 128 256 320 640 1280 5120 8960 12800 16640 20480 24320 28160 32000 35840"
make executable
for n in $N
do
  for bs in $BS
  do
    echo "----- run GEMM $n x $n, BS=$bs" >& $log &   
    CUDA_VISIBLE_DEVICES=0 numactl -m 0 -C 0 ./executable $n $bs >& $log &

    #Move .csv file to directory $dirname
    dir=$prefix/$n/$bs
    mkdir -p $dir
    mv executionTime.csv $dir
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
mv log_file_noCudaGraph.txt $dir2

dir3=${prefix}_records
mkdir $dir3
mv *.csv $dir3

dir4=${prefix}_plots
mkdir $dir4
mv *.pdf $dir4

rm -f executionTime.csv