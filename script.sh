#!/bin/bash

log=log_file.txt
printf "This is the Log File of date" > $log

N="7680 15360 23040 30720 38400"
BS="64 128 192 256 320 384 512 640 768 960 1280 1536 1920 2560 3840 7680"
make executable

timer_start = date "+%Y-%m-%d %H-%M-%S" >> $log

for n in $N
do
  for bs in $BS
  do
    echo "----- run GEMM $n x $n, BS=$bs" >> $log    
    numactl -m 0 -C 0 ./executable $n $bs >> $log

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
  cat $prefix/$n/*/*.csv > ${n}.csv >> $log
  sort -u -n ${n}.csv > ${n}_all.csv >> $log
  ./analysis.r ${n}_all.csv >> $log
done

timer_end = date "+%Y-%m-%d %H-%M-%S" >> $log
duration = `echo eval $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "This experience has took： $duration" >> $log

dir2 = $prefix
mv log_file.txt $dir2

rm -f captureTime.csv
rm -f instantiationTime.csv
rm -f launchingTime.csv