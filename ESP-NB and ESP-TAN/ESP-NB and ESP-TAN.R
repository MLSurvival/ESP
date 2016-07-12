#ESP Algorithm for paper "Early-Stage Event Prediction for Longitudinal Data" presented in PAKDD 2016
#http://link.springer.com/chapter/10.1007%2F978-3-319-31753-3_12 
#This code covers both ESP_NB and ESP_TAN

#The input file should be in [X,Status,Time]
#X is the vectore of featuress
#Status is 1 if event occures and 0 otherwise
#Time is time of event

library(bnlearn)
library(survival)
library(e1071)
library(gplots)
library(ROCR)
library(MASS)
library(discretization)
library(pracma)
library(plyr)

result<-matrix(nrow=1,ncol=12,byrow=T) #to save ESP result


#read data
data <- read.table("C:/Users/Owner/Desktop/M.Sc. CSC/my code for paper/real data/Naive Bayes/breast/breast.csv", header=TRUE, sep=",");
#data <- read.table("...\\data\\data.csv", header=TRUE, sep=",");
qvector<- c(0.50,.9)
num1=dim(data)[2]
q<-quantile(data[ ,num1],qvector)
tc=q[[1]]
tf=q[[2]]
  
#ignoring censored data from first to cutoff point

seq=which(data$time<=tc)
count<-NA
for(i in seq){
  if (data[i,num1-1]=="0"){
    count[i]=i
  }
}
cc=which(count>0)
if(length(cc)>0){
  data=data[-(cc), ]
}
rownames(data)<-NULL
num=dim(data)[1]

mydata1<-data
mydata2<-data

p<-dim(data)[1]
r<-dim(data)[2]
v<-data
for(i in 1:p){
  if(v[i,r]>=tc){
    v[i,r-1]="0"
  }
}

for(i in 1:p){
  if(mydata1[i,r]>=tf){
    mydata1[i,r-1]="0"
  }
}

for(i in 1:p){
  if(mydata2[i,r]>=tf){
    mydata2[i,r-1]="0"
  }
}

#making startified k fold 
require(plyr)
createFolds <- function(x,k){
  n <- nrow(x)
  x$folds <- rep(1:k,length.out = n)[sample(n,n)]
  x
}

folds <- ddply(v,.(status),createFolds,k = 10)
data[c("rname","prob_surv","prob_event","pred_class")] <- NA

data$rname<-rownames(data)

cols<- colnames(v)
v[,cols] <- data.frame(apply(v[cols], 2, as.factor))
#cross validation
f<-dim(folds)[2]
for(k in 1:10){
  test=v[which(folds[ ,f]==k),]
  train=v[which(folds[ ,f]!=k),]
  test=test[ ,-r]
  train=train[ ,-r]
  
  tan = tree.bayes(train, "status")
  fitted = bn.fit(tan, train, method = "bayes")
  pred0 = predict(fitted, test, prob=TRUE)
  #pred_class= pred0
  pred=t(attr(pred0, 'prob'))
  
  #use the below three code for ESP_NB
  #nb <- naiveBayes(as.factor(status)~., data = train,laplace = 1)
  #pred_class<-predict(nb, test)
  #pred<-predict(nb, test,"raw")
  
  for(j in 1: dim(test)[1]){
    a<-as.numeric(rownames(test)[j])
    data$prob_surv[a]<-pred[j,2]
    data$prob_event[a]<-pred[j,1]
    #pp<-as.matrix(pred_class)
    if (data$prob_event[a]>0.5){
      data$pred_class[a]=1
    }
    else  data$pred_class[a]=0
    
  }
}

pred1 <- prediction(data$prob_event,v[ ,r-1])
perf1 <- performance(pred1, "tpr", "fpr")
AUC1 <-attributes(performance(pred1, "auc"))$y.values[[1]]
con1<-table(v[ ,r-1],data$pred_class)
accu1<-(con1[1]+con1[4])/(con1[1]+con1[2]+con1[3]+con1[4])
pre1<- con1[4]/(con1[3]+con1[4])
rec1<- con1[4]/(con1[2]+con1[4])
f1<- 2*pre1*rec1/(pre1+rec1)
g1<- sqrt(pre1*rec1)

#for extrapolation part (ecdf is same as K-M so we need to use linear model for forecasting)
P<-ecdf(data$time)
A<-data$prob_event
x<-P(tc)

y<-(tf/tc)*x
if(y>1){y=1}
#y<-P(tf)
new_prob <- y/(y+(((1-y)*(1-A)*x)/(A*(1-x))))
data$prob_event2<-new_prob
for(o in 1: num){     
  if(new_prob[o]>0.5){
    data$pred_class2[o]=1
  }
  else data$pred_class2[o]=0
}
data2=data
mydata22=mydata2 
#ignoring censored data between the interval of cutoff point and the one of interest
seq=which(tc<=mydata22$time&mydata22$time<=tf)
count<-NA
for(i in seq){
  if (mydata22[i,num1-1]==0){
    count[i]=i
  }
}
cc=which(count>0)
if (length(cc)>0){
  mydata22=mydata22[-(cc), ]     
  data2=data2[-(cc), ]
}


pred2 <- prediction(data2$prob_event2,mydata22$status)
perf2 <- performance(pred2, "tpr", "fpr")
AUC2 <-attributes(performance(pred2, "auc"))$y.values[[1]]
con2<-table(mydata22$status,data2$pred_class2)
accu2<-(con2[1]+con2[4])/(con2[1]+con2[2]+con2[3]+con2[4])
pre2<- con2[4]/(con2[3]+con2[4])
rec2<- con2[4]/(con2[2]+con2[4])
f2<- 2*pre2*rec2/(pre2+rec2)
g2<- sqrt(pre2*rec2)

result[1]=f1;
result[2]=accu1;
result[3]=AUC1;
result[4]=f2;
result[5]=accu2;
result[6]=AUC2;


#for extrapolation part (Weibull)
myfit2 <- fitdistr(data$time[which(data$time<tc)], "weibull",lower = .0001); 
myfit2.shape <- myfit2$estimate[[1]]
myfit2.scale <- myfit2$estimate[[2]]
A <- data$prob_event
x <- 1-exp(-(tc/myfit2.scale)^myfit2.shape)
y <- 1-exp(-(tf/myfit2.scale)^myfit2.shape)
if(y>1){
  y=1
}

new_prob <- y/(y+(((1-y)*(1-A)*x)/(A*(1-x))))
data$prob_event2<-new_prob

for(o in 1: num){
  if(new_prob[o]>0.5){
    data$pred_class2[o]=1
  }
  else data$pred_class2[o]=0
}
data2=data
mydata222=mydata2
#ignoring censored data between the interval of cutoff point and the one of interest
seq=which(tc<=mydata222$time&mydata222$time<=tf)
count<-NA
for(i in seq){
  if (mydata222[i,2]==0){
    count[i]=i
  }
}
cc=which(count>0)
if (length(cc)>0){
  mydata222=mydata222[-(cc), ]     
  data2=data2[-(cc), ]
}

pred2 <- prediction(data2$prob_event2,mydata222$status)
perf2 <- performance(pred2, "tpr", "fpr")
AUC2 <-attributes(performance(pred2, "auc"))$y.values[[1]]
con2<-table(mydata222$status,data2$pred_class2)
accu2<-(con2[1]+con2[4])/(con2[1]+con2[2]+con2[3]+con2[4])
pre2<- con2[4]/(con2[3]+con2[4])
rec2<- con2[4]/(con2[2]+con2[4])
f2<- 2*pre2*rec2/(pre2+rec2)
g2<- sqrt(pre2*rec2)


result[7]=f2;
result[8]=accu2;
result[9]=AUC2;

#for extrapolation part (lognormal)
myfit3 <- fitdistr(data$time[which(data$time<tc)], "lognormal"); 
myfit3.meanlog <- myfit3$estimate[[1]]
myfit3.sdlog <- myfit3$estimate[[2]]
A <- data$prob_event
x <- 0.5+0.5*erf((log2(tc)-myfit3.meanlog)/(sqrt(2)*myfit3.sdlog))
y <- 0.5+0.5*erf((log2(tf)-myfit3.meanlog)/(sqrt(2)*myfit3.sdlog))
if(y>1){
  y=1
}

new_prob <- y/(y+(((1-y)*(1-A)*x)/(A*(1-x))))
data$prob_event2<-new_prob


for(o in 1: num){
  if(new_prob[o]>0.5){
    data$pred_class2[o]=1
  }
  else data$pred_class2[o]=0
}
data2=data
#ignoring censored data between the interval of cutoff point and the one of interest
seq=which(tc<=mydata2$time&mydata2$time<=tf)
count<-NA
for(i in seq){
  if (mydata2[i,2]==0){
    count[i]=i
  }
}
cc=which(count>0)
if (length(cc)>0){
  mydata2=mydata2[-(cc), ]     
  data2=data2[-(cc), ]
}

pred2 <- prediction(data2$prob_event2,mydata2$status)
perf2 <- performance(pred2, "tpr", "fpr")
AUC2 <-attributes(performance(pred2, "auc"))$y.values[[1]]
con2<-table(mydata2$status,data2$pred_class2)
accu2<-(con2[1]+con2[4])/(con2[1]+con2[2]+con2[3]+con2[4])
pre2<- con2[4]/(con2[3]+con2[4])
rec2<- con2[4]/(con2[2]+con2[4])
f2<- 2*pre2*rec2/(pre2+rec2)
g2<- sqrt(pre2*rec2)

result[10]=f2;
result[11]=accu2;
result[12]=AUC2
  
