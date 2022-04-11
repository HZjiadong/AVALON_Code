#!/bin/bash

log=log_file_noCudaGraph.txt
printf "This is the Log File of date" > $log
timer_start=`date +"%Y-%m-%d %H:%M:%S"`
echo "start Time "$timer_start >> $log
N="1280 2560 3840 5120 6400 7680 8960 10240 11520 12800 14080 15360 16640 17920 19200 20480 21760 23040 24320 25600 26880 28160 29440 30720 32000 33280 34560 35840 37120 38400"
BS="64 128 256 320 640 1280"

for n in $N
do
  cat 2022-03-31/$n/*/*.csv > ${n}.csv >> $log
  sort -u -n ${n}.csv > ${n}_all.csv >> $log
  ./tempanalysis.r ${n}_all.csv >> $log
done

timer_end=`date +"%Y-%m-%d %H:%M:%S"`
echo "end Time "$timer_end >> $log

dir3=2022-03-31_records
mv *.csv $dir3

dir4=2022-03-31_records
mv *.pdf $dir4
