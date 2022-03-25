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
# - mean = the mean of time values for each group with same (dimension,blocksize,kernel,operation)
# - sd : the standard devision of the time values (of each group)
# - n : the number of data values in each group
# - min_time/max_time: the min/max of the time values of each group.
# All the stats are computed after removing the values with index =1
makestats<- function(dfraw)
{	
	citol= 0.95
	df <- dfraw %>% filter(index != 1) %>%
	      # make group of data with same value for following column:
	      group_by(dimension,blocksize,kernel,operation) %>% 
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
#df <- readfile(file); 
l <- makestats(df);
print(l);


# theplot
# - X: number of kernels
# - Y: the time for each operation
theplot = ggplot() +
  theme_bw(base_size=16) +
   xlab("#kernels") +
   ylab("Time (s)") +
   scale_fill_brewer(palette = "Set1") +
   theme (
       legend.spacing = unit(.1, "line"),
       panel.grid.major = element_blank(),
       panel.spacing=unit(0, "cm"),
       panel.grid=element_line(size=0),
       legend.position = "bottom",
       legend.title =  element_text("Helvetica")
   ) +
   guides(fill = guide_legend(nrow = 1)) +
   geom_line(data=l, aes(color=operation,
                         x=kernel,
                         y=mean
                        )) +
   scale_y_reverse();
# pdf("gantt.pdf", width=10, height=6)
# print(theplot)
# dev.off()

ggsave(theplot, file="plot.pdf", width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
