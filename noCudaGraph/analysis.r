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
# All the stats are computed after removing the values with index =1
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
#df <- readfile(file); 
l <- makestats(df);
l$tflops <- (2.0*l$dimension*l$dimension*l$dimension/l$mean)*1e-12;
l$tflops[which(l$operation!="launch")]=0;
print(l);


# theplot
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
   geom_line(data=l, aes(x=kernel,
                         y=mean
                        )) +
   geom_point(data=l, aes(x=kernel,
                         y=mean
                        )) +
   facet_grid( ~dimension);

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
   geom_line(data=l, aes(x=blocksize,
                         y=mean
                        )) +
   geom_point(data=l, aes(x=blocksize,
                         y=mean
                        )) +
   facet_grid( ~dimension);
ggsave(theplot2, file="plot2.pdf", width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);

                       
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
   geom_point(data=df, aes(x=blocksize,
                         y=time
                        )) +
   facet_grid( ~dimension);
 
ggsave(theplot1, file=paste("myplot1_",args[1],".pdf",sep=""), width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
ggsave(theplot2, file=paste("myplot2_",args[1],".pdf",sep=""), width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
ggsave(theplot3, file=paste("myplot3_",args[1],".pdf",sep=""), width=29.7/1.2, height=42/1.2/3, units="cm", dpi=300);
