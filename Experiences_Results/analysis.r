#!/usr/bin/env Rscript

library(dplyr);
library(ggplot2);


# Read csv file and return a data frame. Format of column are described in the first row.
readfile <- function (filename)
{
   df <- read.csv(paste("",filename,sep=''), header=TRUE, sep=",", strip.white=TRUE);
   df;
}

#Merge several measures of the same point together
# Return a data frame with columns :
# - mean = the mean of time values for each group with same (dimension,blocksize,kernel)
# - sd : the standard devision of the time values (of each group)
# - n : the number of data values in each group
# - min_time/max_time: the min/max of the time values of each group.
# All the stats are computed after removing the values with index =1, in case of avoiding iregular initiation value
makestats<- function(dfraw)
{	
	citol= 0.95
	df <- dfraw %>% filter(index != 0) %>%
	      # make group of data with same value for following column:
	      group_by(dimension,blocksize,kernel) %>% 
	      # summarize goup of data by computing time= mean of time of each group, etc
	      summarize( 
	          n=length(time),
	          sd=sd(time),
	          min_time=min(time), 
	          max_time=max(time),
	          mean=mean(time)
	       );
	df$ci <- qt(citol + (1 - citol)/2, df$n - 1) * df$sd/sqrt(df$n);
	df<-	as.data.frame(df);
	df;
} 

# Read file in command lines
args <- commandArgs(trailingOnly=TRUE)
df <- readfile(args[1]);

#When using with interactive R, I define file="" to point to my file then I uncomment the next line to read the file.
#And at the end I source the file analysis.R: source("analysis.R")
# - flop: number of floating point calculation per second, order of magnitude of "Tera"(10^12)
#df <- readfile(file); 
l <- makestats(df);
l$tflops <- (2.0*l$dimension*l$dimension*l$dimension/l$mean)*1e-12;
l$cudagraph <- 0;
print(l);

