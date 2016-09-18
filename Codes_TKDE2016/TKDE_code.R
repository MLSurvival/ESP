#ESP Algorithm for paper "A Bayesian Perspective on Early Stage Event Prediction in Longitudinal Data"
#http://dmkd.cs.vt.edu/papers/TKDE17.pdf

#The input file should be in [X,Status,Time]
#X is the vectore of featuress
#Status is 1 if event occures and 0 otherwise
#Time is time of event

##Code consists of three steps:
# 1-preprocessing
# 2-model building
# 3-postprocessing
#This code covers step 1 and 3. Step 2 can be done in other softwares, such as Weka.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##preprocessing data

##load library
library(OIsurv)
library(survival)
library(MASS)
library(ROCR)
library(fitdistrplus)
library(actuar)

##load data
data <- read.table("C:\\data.csv", header=TRUE, sep=",")


#count the number of all events
seq=which(data$status==1)
en=0
for(i in seq){
  en = en +1
}
cens = dim(data)[1]-en
cat("The number of all events is: ", en)
cat("The number of all censorings is: ", cens)

#the number of 50% events
enh = ceiling(en*0.50) 
cat("The number of 50% events is: ", enh)

#find the cutoff point for 50% events
n=1
for(i in seq){
  if(n<=enh){
    time = data$time[i]
    n = n+1
  }
}
tc<-time
cat("The cutoff point is: time =", tc)

#count the number of samples before tc
seq=which(data$time<=tc)
s=0
for(i in seq){
  s = s +1
}
cat("The number of samples before tc is: ", s)

#the number of censoring before tc
seq=which(data$status==0 & data$time<=tc)
cn=0
for(i in seq){
  cn = cn +1
}
cat("The number of censoring before tc is: ", cn)
cat("The ratio of censoring before tc is: ", cn/s)

#ignoring censored data before the cutoff point
count<-NA
for(i in seq){
  count[i]=i
}
cc=which(count>0)
if (length(cc)>0){
  data=data[-(cc), ]
}

#normalization of the features
normalize <- function(x) { 
  x <- as.matrix(x)
  minAttr=apply(x, 2, min)
  maxAttr=apply(x, 2, max)
  x <- sweep(x, 2, minAttr, FUN="-") 
  x=sweep(x, 2,  maxAttr-minAttr, "/") 
  attr(x, 'normalized:min') = minAttr
  attr(x, 'normalized:max') = maxAttr
  return (x)
} 
d=dim(data)[2]
data[3:d]<-normalize(data[,3:d])

#add "id" as a column in the dataset
data$id <- NA
data$rand <- runif(dim(data)[1], 0.0, 1.0)
data <- data[order(data$rand),] 
data$id <- c(1:dim(data)[1])
data <- data[,1:dim(data)[2]-1]

#reorder the columns
data0 <- data[,-c(1,2)]
data00 <- data[, c("id", "time")]
data000 <- data[, c("id", "status")]
data <- merge(data00, data0, by="id")
data <- merge(data, data000, by="id")

#create 100% dataset
write.csv(data,"C:\\data_100.csv")

##create 50% dataset
#set the status of the events after tc to be censored
#data <- data[order(data$time),] #to checking the results
for(i in 1:dim(data)[1]){
  if(data$time[i]>tc & data$status[i]==1){
    data$status[i]=0
  }
}
for(i in 1:dim(data)[1]){
  if(data$time[i]>tc){
    data$time[i]=tc
  }
}

write.csv(data,"C:\\data_50.csv")


#load 100% event dataset
data <- read.csv("C:\\data_100.csv", 1)
data <- data[,-1]

##computing the weights 
#add "censor" as a column
data$censor <- 1 - data$status

data1 <- data[,-c(1,dim(data)[2]-1)] #this is time, censor and rest attirbutes
data2 <- data[,-c(1,dim(data)[2])] #this is time status and rest attirbutes
d0 <- dim(data)[1]

#kaplan meier method
if(TRUE){
  #gives probability of "not being censored"
  my.surv <- Surv(data1$time, data1$censor)
  my.fit <- survfit(my.surv~1)
  w <- my.fit$surv
  survtime <- my.fit$time
  
  v <- NA
  for (i in 1:d0){
    for(j in 1:length(w)){
      if(data1$time[i] == survtime[j]){
        v[i] <- w[j]
      }
    }
  }
  
  weight.cen <- v
  
  #gives probability of "survival"
  my.surv <- Surv(data2$time, data2$status)
  my.fit <- survfit(my.surv~1)
  w <- my.fit$surv
  survtime <- my.fit$time
  
  v <- NA
  for (i in 1:d0){
    for(j in 1:length(w)){
      if(data2$time[i] == survtime[j]){
        v[i] <- w[j]
      }
    }
  }
  
  weight <- v
  myweight <- data.frame(weight=weight, weight_cen=weight.cen)
  }

##add "prob_event" and "prob_eventfree" as new columns and assign new labels
data$prob_event <- 1 - weight
data$prob_eventfree <- 1- weight.cen

#label new status
for(i in 1:dim(data)[1]){
  if(data$status[i]==0){
    if(data$prob_event[i] >= data$prob_eventfree[i]){
      data$status[i] <- 1
    }
    else if(data$prob_event[i] < data$prob_eventfree[i]){
      data$status[i] <- 0
    }
  }
  else{
    data$status[i] <- 1
  }
}
status_100 <- data$statu
data <- data[, -c(dim(data)[2]-2, dim(data)[2]-1,dim(data)[2])]
data <- data[order(data$time),] 
write.csv(data,"C:\\data_100_new.csv")

data <- read.csv("C:\\data_50.csv", 1)
data <- data[,-1]

data$status_100 <- status_100
write.csv(data,"C:\\data_50.csv")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##model building
#for this step we use data_50 and data_100 created in preprocessing step to build a classifier using
#naive bayes, TAN or bayesian network. We use Weka for this step

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##postprocessing data

#input:
#data_50 = read 50%-event data after doing weka and adding prob_event_weka
#data_100 = read 100%-event original data after relabeling

#based on the which distribution we choose for the extrapolation part of ESP algorithm
ex = FALSE
we = FALSE
log = TRUE

##load data
data_50 <- read.table("C:\\data_50.csv", header=TRUE, sep=",")
data_100 <- read.table("C:\\data_100_new.csv", header=TRUE, sep=",")

data_50_aft <- data_50[ , -which(names(data_50) %in% c("id","status_100","prob_event_weka"))]

#reorder the columns
#data0 <- data_50[,-c(2, dim(data_50)[2]-1)]
#data00 <- data_50[, c("id", "time", "status")]
#data_50 <- merge(data00, data0, by="id")

#delete column "id"
#data_50 <- data_50[, -1]
#data_100 <- data_100[, -1]

###################################Estimating Time ######################################
#fitting aft to data uo to tc (50% event data)
#exponential
if(ex){
  aft<-survreg(Surv(data_50_aft$time, data_50_aft$status) ~ ., data_50_aft,dist="exponential") 
  
  #find Time of event estimated by aft model
  coef1<-summary(aft)$coef
  aft_coef <- NA
  for (i in 1:length(coef1)){
    aft_coef[i]=coef1[[i]]
  }
  x<-data_50_aft[  ,-c(1,2)]
  x <- as.matrix(x)
  aft_coef<- as.matrix(aft_coef)
  w <- x%*%aft_coef[-1]
  ww <- w-aft_coef[1]
  Time <- exp(ww)
  
}

#weibull
if(we){
  aft<-survreg(Surv(data_50_aft$time, data_50_aft$status) ~ ., data_50_aft,dist="weibull") 
  
  #find Time of event estimated by aft model
  coef1<-summary(aft)$coef
  aft_coef <- NA
  for (i in 1:length(coef1)){
    aft_coef[i]=coef1[[i]]
  }
  x<-data_50_aft[  ,-c(1,2)]
  x <- as.matrix(x)
  aft_coef<- as.matrix(aft_coef)
  w <- x%*%aft_coef[-1]
  ww <- w+aft_coef[1]
  Time <- exp(ww)
}

#loglogistic
if(log){
  aft<-survreg(Surv(data_50_aft$time, data_50_aft$status) ~ ., data_50_aft,dist="loglogistic") 
  
  #find Time of event estimated by aft model
  coef1<-summary(aft)$coef
  aft_coef <- NA
  for (i in 1:length(coef1)){
    aft_coef[i]=coef1[[i]]
  }
  x<-data_50_aft[  ,-c(1,2)]
  x <- as.matrix(x)
  aft_coef<- as.matrix(aft_coef)
  w <- x%*%aft_coef[-1]
  ww <- w+aft_coef[1]
  Time <- exp(ww)
}

#####################################Extrapolation######################################
#if use exponential in the aft model (survreg) use exponential here and vs for weibull
#####################################################
#num=dim(data_result)[1] #??????????????
num = dim(data_50)[1]
tc <- max(data_50$time)
tf <- max(data_100$time)

##################exponential#################
if(ex){
  myfit2 <- fitdistr(Time, "exponential"); 
  myfit2.mean <- myfit2$estimate[[1]]
  A <- data_50$prob_event_weka
  x <- 1-exp(-myfit2.mean*tc)
  y <- 1-exp(-myfit2.mean*tf)
  if(y>1){
    y=1
  }
}

##################weibull#################
if(we){
  myfit4 <- fitdistr(Time, "weibull",lower = .0001); 
  myfit4.shape <- myfit4$estimate[[1]]
  myfit4.scale <- myfit4$estimate[[2]]
  A <- data_50$prob_event_weka
  x <- 1-exp(-(tc/myfit4.scale)^myfit4.shape)
  y <- 1-exp(-(tf/myfit4.scale)^myfit4.shape)
  if(y>1){
    y=1
  }
}

#################loglogistic################
if(log){
  TTime<-as.numeric(Time)
  myfit5 <- fitdist(TTime, "llogis",start = list(shape = 1, scale = 1)); 
  myfit5.shape <- myfit5$estimate[[1]]
  myfit5.scale <- myfit5$estimate[[2]]
  A <- data_50$prob_event_weka
  x <- 1-exp(-(tc/myfit5.scale)^myfit5.shape)
  y <- 1-exp(-(tf/myfit5.scale)^myfit5.shape)
  if(y>1){
    y=1
  }
}

new_prob_event <- y/(y-(((1-y)*(1-A)*x)/(A*(1-x))))

data_50$prob_new_event <- new_prob_event #basically we don't need this cloumn in 50% event data so we can add it to 100%- event data

#from now we are only dealing with 100%-event dataset
#in 100%event data we have label column we need to add new_status column based on the new_prob_event
#if new_prob_event>0.5 then new_status=1 and 0 otherwise
for(i in 1:dim(data_50)[1]){
  if(data_50$prob_new_event[i] >= 0.1){
    data_50$new_status[i] <- 1
  }
  else {
    data_50$new_status[i] <- 0
  }
}

write.csv(data_50,"C:\\data_50_predict.csv")

##################################################evaluation measurment#########################################
con <- table(data_50$status_100, data_50$new_status)
con
pred <- prediction(data_50$prob_new_event, data_50$status_100)
perf <- performance(pred, "tpr", "fpr")
AUC <- attributes(performance(pred, "auc"))$y.values[[1]]
AUC
accu <- (con[1]+con[4])/(con[1]+con[2]+con[3]+con[4])
accu
pre_1 <- con[4]/(con[3]+con[4])
rec_1 <- con[4]/(con[2]+con[4])
pre_0 <- con[1]/(con[2]+con[1])
rec_0 <- con[1]/(con[3]+con[1])

f_1 <- 2*pre_1*rec_1/(pre_1+rec_1)
f_0 <- 2*pre_0*rec_0/(pre_0+rec_0)
f_measure <- (f_1*(con[3]+con[4])+f_0*(con[1]+con[2]))/(con[1]+con[2]+con[3]+con[4])
f_measure

