#Copy all the missclasified frames

remaining<-dat %>% filter(Truth=="Positive",label=="Negative",Binary < 0.1)

basename<-"/Users/Ben/DeepMeerkat/DeepMeerkat_20180301_145641"
destbase<-""
for(x in 1:nrow(remaining)){
  filen<-paste(basename,"/",remaining$Video[x],"/",remaining$Frame[x],".jpg",sep="")
  destname<-"/Users/ben/Dropbox/DeepLearning/Misclassified"
  file.copy(filen,destname)
}

#Copy all the best frames

remaining<-dat %>% filter(Truth=="Positive",label=="Positive",Binary > 0.9)

basename<-"/Users/Ben/DeepMeerkat/DeepMeerkat_20180301_145641"
destbase<-""
for(x in 1:nrow(remaining)){
  filen<-paste(basename,"/",remaining$Video[x],"/",remaining$Frame[x],".jpg",sep="")
  destname<-"/Users/ben/Dropbox/DeepLearning/Classified/"
  file.copy(filen,destname)
}