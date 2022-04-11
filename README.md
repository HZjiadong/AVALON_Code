# AVALON_Code
Code that used during PFE AVALON INRIA

noCudaGraph Folder:
Contain all files needed for experience without CUDA Graph structure

yesCudaGraph Folder:
Contain all files needed for experience with CUDA Graph structure

Main.cpp: 
main program of CUDA Graph API used inside.
certains code source: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
  
flops.h:
header file contains all formulas for Flops( floating points operation per second) calculation.

Makefile:
used by script.sh
also used for unite test for one pair of (Dimension,Blocksize).

analysis.r:
using experience records to plot all graphical results needed for analyzation

script.sh:
Automate script for the following uses:
1. launch all pair of (Dimension,Blocksize) exercise using Makefile.
2. create respectly catalogues for data stock.
3. sum all records and plot graphic results using analysis.r.

