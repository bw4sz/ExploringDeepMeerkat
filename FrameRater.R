library(dplyr)
library(ggplot2)

a<-list.files("/Users/Ben/Dropbox/HummingbirdProject/",pattern="parameters.csv",recursive=T,full.names=T)
b<-bind_rows(lapply(a,read.csv,header=F))
d<-b %>% filter(V1=="Frame processing rate")
d$rate<-as.numeric(d$V2)
quantile(d$rate)
qplot(d$rate)
