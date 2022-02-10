# AVALON_Code
Code that used during PFE AVALON INRIA


randomMaker.cpp:
Compile:g++ -std=c++11 randomMaker.cpp -o randomMaker
Execute:./randomMaker <dimension>
实测单线程生成10000*10000的随机数矩阵只要2s
核心思想在于将二维矩阵转化为一维数组进行操作



Main Code source: 

https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/

