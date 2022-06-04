#!/bin/bash

log=log_file.txt
printf "This is the Log File of date" > $log
timer_start=`date +"%Y-%m-%d %H:%M:%S"`
echo "start Time "$timer_start >> $log
prefix=`date +"%Y-%m-%d"`
N="6400 12800 25600"
BS="640 3200 6400 "
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
    mv createCudaGraphExplicitDependency.csv $dir
    mv instantiateCudaGraphExplicitDependency.csv $dir
    mv executeCudaGraphExplicitDependency.csv $dir
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

rm -f createCudaGraphExplicitDependency.csv
rm -f instantiateCudaGraphExplicitDependency.csv
rm -f executeCudaGraphExplicitDependency.csv.csv