library(class)
library(reshape2)
library(ggplot2)
library(leaps)
library(glmnet)


x.train<-rbind(train2,train3)
y.train<-matrix(NA,nrow = 1389,ncol=1)
y.train[1:731]<-rep(0,731)
y.train[732:1389]<-rep(1,658)
y.train1<-y.train
y.train<-y.train == 1

x.test<-rbind(test2,test3)
y.test<-matrix(NA,nrow = 364,ncol=1)
y.test[1:198]<-rep(0,198)
y.test[199:364]<-rep(1,166)
y.test1<-y.test
y.test<-y.test == 1

######### lm
lm <- lm(y.train ~ x.train)
yhat <- (predict(lm,as.data.frame(x.train))) > 0.5
le.train<-mean(yhat != y.train)
y.test.hat<-(cbind(1,x.test) %*% coef(lm)) > 0.5
le.test <- mean(y.test.hat != y.test)

######### knn
library(class)
k.vec<-c(1,3,5,7,15)
n<-length(k.vec)
test.error<-rep(NA,n)
train.error<-rep(NA,n)
for(i in 1:n){
  yhat<-knn(x.train,rbind(x.train,x.test),y.train,k=k.vec[i],prob = FALSE)
  train.error[i]<-mean(yhat[1:1389] != y.train)
  test.error[i] <- mean(yhat[1390:1753] != y.test)
}

######### plot
library(reshape2)
library(ggplot2)
df<- data.frame("k"<-k.vec, 
                "kNN.Train"<-train.error,"kNN.Test"<-test.error,
                "LR.Train"<-le.train,"LR.Test"<-le.test)
plot.data <- melt(df, id<-"k") 
ggplot(data=plot.data,aes(x=k, y=value, colour=variable)) +
  geom_line() +geom_point() +xlab("k") + ylab("classifying error") +
  ggtitle("Classification Errors") +theme(plot.title = element_text(hjust = 0.5)) + 
  scale_colour_hue(name="Method",
                   labels=c("kNN.Train","k-NN.Test","LR.Train", "LR.Test"))

cbind(k.vec,test.error,train.error)
list(LR.train = le.train,LR.test = le.test)


######### best subset selection
library(leaps)
train<-cbind(as.data.frame(y.train),as.data.frame(x.train))
colnames(train)[1]<-"y"
reg.full<-regsubsets(y ~ .,train,really.big=TRUE)

reg.full3<-regsubsets(y ~ .,train,nvmax=3,really.big=TRUE)
outm<-summary(reg.full3)$outmat
which(outm[3,]=="*")
# "nvmax" means the	maximum size of subsets to examine, 
# which helps us to return as many as variables that we desire.
# The best model with only 3 variables retains 104,166,249 variables.


######### forward selection
regfit.fwd<-regsubsets(y~.,train,method="forward")
omfwd<-summary(regfit.fwd)$outmat
# which(omfwd[9,]=="*")
# We cannot find the variables retained in the model with 9 variables. 
# There are only maximum 3 variables in the model.

regfit.fwd2<-regsubsets(y~.,train,method="forward",nvmax = 256)
omfwd2<-summary(regfit.fwd2)$outmat
which(omfwd2[9,]=="*")
which(omfwd2[3,]=="*")
# Here are 9 variabels when we consider models with 9 predictors

# show the pictures of RSS, adjusted RSq, Cp, and BIC against the number of variables. 
# Color the points that should be chosen according to 3 criteria, adjusted RSq, Cp, and BIC. 
par(mfrow=c(2,2))
plot(summary(regfit.fwd2)$rss,xlab="Number of variables", ylab="RSS", type = "l")
plot(summary(regfit.fwd2)$adjr2,xlab="Number of Variables", ylab="Adjusted RSq", type="l")
points(which.max(summary(regfit.fwd2)$adjr2),summary(regfit.fwd2)$adjr2[which.max(summary(regfit.fwd2)$adjr2)],col="red",cex=2,pch=20)
plot(summary(regfit.fwd2)$cp,xlab="Number of Variables", ylab="Cp", type='l')
points(which.min(summary(regfit.fwd2)$cp),summary(regfit.fwd2)$cp[which.min(summary(regfit.fwd2)$cp)],col="red",cex=2,pch=20)
plot(summary(regfit.fwd2)$bic,xlab="Number of Variables", ylab="BIC", type='l')
points(which.min(summary(regfit.fwd2)$bic),summary(regfit.fwd2)$bic[which.min(summary(regfit.fwd2)$bic)],col="red",cex=2,pch=20)
which.max(summary(regfit.fwd2)$adjr2)
which.min(summary(regfit.fwd2)$cp)
which.min(summary(regfit.fwd2)$bic)

# If we use the best model according to Cp or bic from the forward selection
# training and testing errors
#fwd cp
yhat.fwd <- (cbind(1,x.train[,c(names(coef(regfit.fwd2,75))[-1])]) %*% coef(regfit.fwd2,75)) > 0.5
trainfwd<-mean(yhat.fwd != y.train)
ytest.fwd<-(cbind(1,x.test[,c(names(coef(regfit.fwd2,75))[-1])]) %*% coef(regfit.fwd2,75)) > 0.5
testfwd <- mean(ytest.fwd != y.test)
list(cp_training_fwd = trainfwd,
     cp_testing_fwd = testfwd)

#fwd bic
yhat.fwd2 <- (cbind(1,x.train[,c(names(coef(regfit.fwd2,34))[-1])]) %*% coef(regfit.fwd2,34)) > 0.5
trainfwd2<-mean(yhat.fwd2 != y.train)
ytest.fwd2<-(cbind(1,x.test[,c(names(coef(regfit.fwd2,34))[-1])]) %*% coef(regfit.fwd2,34)) > 0.5
testfwd2 <- mean(ytest.fwd2 != y.test)
list(bic_training_fwd = trainfwd2,
     bic_testing_fwd = testfwd2)




######### backward selection
regfit.bwd<-regsubsets(y~.,train,method="backward",nvmax = 256)
ombwd<-summary(regfit.bwd)$outmat
which(ombwd[3,]=="*")
# We find that the 3 variables are 104,166,249.
# By comparing 3 models we got, we find the model containing 3 variables is same, 
# which also has 2 variables retained in the model with 9 variables. 

par(mfrow=c(2,2))
plot(summary(regfit.bwd)$rss,xlab="Number of variables", ylab="RSS", type = "l")
plot(summary(regfit.bwd)$adjr2,xlab="Number of Variables", ylab="Adjusted RSq", type="l")
points(which.max(summary(regfit.bwd)$adjr2),summary(regfit.bwd)$adjr2[which.max(summary(regfit.bwd)$adjr2)],col="red",cex=2,pch=20)
plot(summary(regfit.bwd)$cp,xlab="Number of Variables", ylab="Cp", type='l')
points(which.min(summary(regfit.bwd)$cp),summary(regfit.bwd)$cp[which.min(summary(regfit.bwd)$cp)],col="red",cex=2,pch=20)
plot(summary(regfit.bwd)$bic,xlab="Number of Variables", ylab="BIC", type='l')
points(which.min(summary(regfit.bwd)$bic),summary(regfit.bwd)$bic[which.min(summary(regfit.bwd)$bic)],col="red",cex=2,pch=20)

# show the pictures of RSS, adjusted RSq, Cp, and BIC against the number of variables. 
# Color the points that should be chosen according to 3 criteria, adjusted RSq, Cp, and BIC. 
which.max(summary(regfit.bwd)$adjr2)
which.min(summary(regfit.bwd)$cp)
which.min(summary(regfit.bwd)$bic)

# If we use the best model according to Cp or bic from the backward selection
# training and testing errors
#bwd cp
yhat.bwd <- (cbind(1,x.train[,c(names(coef(regfit.bwd,89))[-1])]) %*% coef(regfit.bwd,89)) > 0.5
trainbwd<-mean(yhat.bwd != y.train)
ytest.bwd<-(cbind(1,x.test[,c(names(coef(regfit.bwd,89))[-1])]) %*% coef(regfit.bwd,89)) > 0.5
testbwd <- mean(ytest.bwd != y.test)
list(cp_training_bwd = trainbwd,
     cp_testing_bwd = testbwd)

#bwd bic
yhat.bwd2 <- (cbind(1,x.train[,c(names(coef(regfit.bwd,38))[-1])]) %*% coef(regfit.bwd,38)) > 0.5
trainbwd2<-mean(yhat.bwd2 != y.train)
ytest.bwd2<-(cbind(1,x.test[,c(names(coef(regfit.bwd,38))[-1])]) %*% coef(regfit.bwd,38)) > 0.5
testbwd2 <- mean(ytest.bwd2 != y.test)
list(bic_training_bwd = trainbwd2,
     bic_testing_bwd = testbwd2)


############ ridge regression with tuning parameter grid
library(glmnet)
lambda.grid <- 10^seq(4,-3,length=100)
ridge.mod <- glmnet(x.train,y.train1,alpha=0,lambda=lambda.grid,standardize=FALSE)
coeff.matrix <- coef(ridge.mod)
plot(coeff.matrix[2,],ylim=c(min(coeff.matrix[-1,]),max(coeff.matrix[-1,])),col=2,type="l",xlab="lambda",ylab="coefficient" )
for(i in 2:257){ 
  lines(coeff.matrix[i,],col=i,type="l")
}

v<-which.min(coeff.matrix[-1,100])
plot(coeff.matrix[2,],ylim=c(min(coeff.matrix[-1,]),max(coeff.matrix[-1,])),col=2,type="l", xlab="lambda",ylab="coefficient")
for(i in 2:129) lines(coeff.matrix[i,],col=i,type="l")
for(i in 131:257) lines(coeff.matrix[i,],col=i,type="l")

yhat.r0<-predict(ridge.mod,s=0,newx=x.train)
trainr0<-mean((yhat.r0 - y.train1)^2)
ytest.r0<-predict(ridge.mod,s=0,newx = x.test)
testr0 <- mean((ytest.r0 - y.test1)^2)
list(training_r0 = trainr0,
     testing_r0 = testr0)

co.r<-predict(ridge.mod,type="coefficient",s=0)
co.l<-summary(lm)$coef[,1]
plot(co.r[-c(1,17)],col="red",type="l",xlab="lamda",ylab="coefficient estimate")
points(co.l[-c(1,17)],col="blue",type="l")

trainerr<-rep(NA,100)
for(i in 1:100){
  ridge.pred<-(predict(ridge.mod,s=lambda.grid[i],newx=x.train))
  trainerr[i]<-mean((ridge.pred - y.train1)^2)
}
testerr<-rep(NA,100)
for(i in 1:100){
  ridge.pred<-(predict(ridge.mod,s=lambda.grid[i],newx=x.test))
  testerr[i]<-mean((ridge.pred - y.test1)^2)
}
par(mfrow = c(1,2))
plot(log(lambda.grid),trainerr,ylim = c(min(trainerr),max(trainerr)))
plot(log(lambda.grid),testerr,ylim = c(min(testerr),max(testerr)))



###### lasso regression with tuning parameter grid
lambda.grid <- 10^seq(0,-5,length=100)
lasso.mod <- glmnet(x.train,y.train1,alpha=1,lambda=lambda.grid,standardize=FALSE)
coeff.matrix <- coef(lasso.mod)
plot(coeff.matrix[2,],ylim=c(min(coeff.matrix[-1,]),max(coeff.matrix[-1,])),col=2,type="l",xlab="lambda",ylab="coefficient")
for(i in 2:257) lines(coeff.matrix[i,],col=i,type="l")

v<-which.max(coeff.matrix[-1,100])
plot(coeff.matrix[2,],ylim=c(min(coeff.matrix[-1,]),max(coeff.matrix[-1,])),col=2,type="l",xlab="lambda",ylab="coefficient")
for(i in 2:32){
  lines(coeff.matrix[i,],col=i,type="l")
}
for(i in 34:257){
  lines(coeff.matrix[i,],col=i,type="l")
}

yhat.l0<-predict(lasso.mod,s=0,newx=x.train)
trainl0<-mean((yhat.l0 - y.train1)^2)
ytest.l0<-predict(lasso.mod,s=0,newx = x.test)
testl0 <- mean((ytest.l0 - y.test1)^2)
list(training_l0 = trainl0,
     testing_l0 = testl0)

co.r<-predict(lasso.mod,type="coefficient",s=0)
co.l<-summary(lm)$coef[,1]
plot(co.r[-c(1,17)],col="red",type="l",xlab="lamda",ylab="coefficient estimate")
points(co.l[-c(1,17)],col="blue",type="l")

trainerr<-rep(NA,100)
for(i in 1:100){
  lasso.pred<-(predict(lasso.mod,s=lambda.grid[i],newx=x.train))
  trainerr[i]<-mean((lasso.pred - y.train1)^2)
}
testerr<-rep(NA,100)
for(i in 1:100){
  lasso.pred<-(predict(lasso.mod,s=lambda.grid[i],newx=x.test))
  testerr[i]<-mean((lasso.pred - y.test1)^2)
}
par(mfrow = c(1,2))
plot(log(lambda.grid),trainerr,ylim = c(min(trainerr),max(trainerr)))
plot(log(lambda.grid),testerr,ylim = c(min(testerr),max(testerr)))








