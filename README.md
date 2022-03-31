# AVALON_Code
Code that used during PFE AVALON INRIA

Main.cpp: 
main program of CUDA Graph API used inside.
code source: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/

with two version:
1. version without cuda graph structure
2. version with cuda graph structure
  
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

