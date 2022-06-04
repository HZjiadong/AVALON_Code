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
# All the stats are computed after removing the values with index =1, in case of avoiding iregular initiation value
makestats<- function(dfraw)
{	
	citol= 0.95
	df <- dfraw %>% filter(index != 0) %>%
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

#When using with interactive R, 
#I define file="" to point to my file then I uncomment the next line to read the file.
#And at the end I source the file analysis.R: source("analysis.R")
# - flop: number of floating point calculation per second, order of magnitude of "Tera"(10^12)
#df <- readfile(file); 
l <- makestats(df);
l$tflops <- (2.0*l$dimension*l$dimension*l$dimension/l$mean)*1e-12;
l$tflops[which(l$operation!="launch")]=0;
print(l);


#Plots:

# theplot1: line chart
# - X: number of kernels
# - Y: the time for each operation
theplot1 = ggplot() +
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
   geom_point(data=l, aes(color=operation,
                        x=kernel,
                        y=mean
                        )) +
   facet_grid( ~dimension);

# theplot2: line chart 
# - X: block size
# - Y: the time for each operation
theplot2 = ggplot() +
  theme_bw(base_size=16) +
   xlab("#blocksize") +
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
                         x=blocksize,
                         y=mean
                        )) +
   geom_point(data=l, aes(color=operation,
                         x=blocksize,
                         y=mean
                        )) +
   facet_grid( ~dimension);

# theplot3: scatter plot
# - X: block size
# - Y: the time for each operation                       
theplot3 = ggplot() +
  theme_bw(base_size=16) +
   xlab("#blocksize") +
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
   geom_point(data=df, aes(color=operation,
                         x=blocksize,
                         y=time
                        )) +
   facet_grid( ~dimension);

# theplot4: line chart
# - X: number of kernels
# - Y: total tera-flops for each operation
theplot4 = ggplot() +
  theme_bw(base_size=16) +
   xlab("#kernels") +
   ylab("TFlop/s") +
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
                         y=tflops
                        )) +
   geom_point(data=l, aes(color=operation,
                         x=kernel,
                         y=tflops
                        )) +
   facet_grid( ~dimension);

# theplot5: scatter plot
# - X: number of kernels
# - Y: total tera-flops for each operation
theplot5 = ggplot() +
  theme_bw(base_size=16) +
   xlab("#blocksize") +
   ylab("Tflop/s") +
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
                         x=blocksize,
                         y=tflops
                        )) +
   geom_point(data=l, aes(color=operation,
                         x=blocksize,
                         y=tflops
                        )) +
   facet_grid( ~dimension);

#Output files:
ggsave(theplot1, file=paste("time_plot1_",args[1],".pdf",sep=""), width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
ggsave(theplot2, file=paste("time_plot2_",args[1],".pdf",sep=""), width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
ggsave(theplot3, file=paste("time_plot3_",args[1],".pdf",sep=""), width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
ggsave(theplot4, file=paste("tflops_plot1_",args[1],".pdf",sep=""), width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
ggsave(theplot5, file=paste("tflops_plot2_",args[1],".pdf",sep=""), width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
