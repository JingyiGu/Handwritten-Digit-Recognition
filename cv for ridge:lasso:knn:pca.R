library(grid)
library(boot)
library(glmnet) 

##### pca
# plots for the second score vs the first score, 
# image of the mean of 3’s
# image of the first loading vector
# image of the second loading vector

tr3pca<-prcomp(train3,scale=FALSE)
x3.mean<-apply(train3,2,mean)
x3.sd<-apply(train3,2,sd)
x3.svd<-svd(train3)
x3.score1<-train3 %*% x3.svd$v
x3.score2<-x3.svd$u %*% diag(x3.svd$d)
plot(x3.score1[,1],x3.score1[,2])

library(grid)
drawDigit <- function(x) {
  for (i in 1:16) {
    for (j in 1:16) {
      color <- gray(1 - (1 + x[(i - 1) * 16 + j])/2)
      grid.rect(j, 17 - i, 1, 1, default.units = "native",
                gp = gpar(col = color, fill = color))
    }
  }
}

grid.newpage()
pushViewport(viewport(xscale = c(0, 6), yscale = c(0, 6)))

pushViewport(viewport(x=1,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr3pca$center)
popViewport(1)

pushViewport(viewport(x=2,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr3pca$rotation[,1])
popViewport(1)

pushViewport(viewport(x=3,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr3pca$rotation[,2])
popViewport(1)

# image of the mean of 2’s
# image of the first loading vector
# image of the second loading vector
tr2pca<-prcomp(train2,scale=FALSE)
x2.mean<-apply(train2,2,mean)
x2.sd<-apply(train2,2,sd)
x2.svd<-svd(train2)
x2.score1<-train2 %*% x2.svd$v
x2.score2<-x2.svd$u %*% diag(x2.svd$d)
plot(x2.score1[,1],x2.score1[,2])

grid.newpage()
pushViewport(viewport(xscale = c(0, 6), yscale = c(0, 6)))

pushViewport(viewport(x=1,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr2pca$center)
popViewport(1)

pushViewport(viewport(x=2,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr2pca$rotation[,1])
popViewport(1)

pushViewport(viewport(x=3,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr2pca$rotation[,2])
popViewport(1)


x.train <- rbind(train2,train3)
tr.pca<-prcomp(x.train,scale=FALSE)
x.mean<-apply(x.train,2,mean)
x.sd<-apply(x.train,2,sd)
x.svd<-svd(x.train)
x.score1<-x.train %*% x.svd$v
x.score2<-x.svd$u %*% diag(x.svd$d)
par(mfrow=c(2,2))
plot(x.score1[1:731,1],col="red")
points(x.score1[732:1389,1],col="green")
plot(x.score1[1:731,2],col="red")
points(x.score1[732:1389,2],col="green")
plot(x.score1[1:731,1],x.score1[1:731,2],col="red")
points(x.score1[732:1389,1],x.score1[732:1389,2],col="green")

grid.newpage()
pushViewport(viewport(xscale = c(0, 6), yscale = c(0, 6)))

pushViewport(viewport(x=1,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr.pca$center)
popViewport(1)

pushViewport(viewport(x=2,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr.pca$rotation[,1])
popViewport(1)

pushViewport(viewport(x=3,y=5,width = 1, height = 1, xscale = c(0, 17), yscale = c(0, 17), default.units = "native"))
drawDigit(tr.pca$rotation[,2])
popViewport(1)


##### cv for ols

set.seed(1)
x.train<-rbind(train2,train3)
y.train<-matrix(NA,nrow = 1389,ncol=1)
y.train[1:731]<-rep(0,731)
y.train[732:1389]<-rep(1,658)
y.train1 <- y.train ==1


x.test<-rbind(test2,test3)
y.test<-matrix(NA,nrow = 364,ncol=1)
y.test[1:198]<-rep(0,198)
y.test[199:364]<-rep(1,166)
y.test1 <- y.test ==1

train<-cbind(as.data.frame(y.train),as.data.frame(x.train))
colnames(train)[1]<-"y"

library(boot)
glm.fit <- glm(y~.,data=train)
cv.err<-cv.glm(train,glm.fit)
cv.err$delta

cost = function(y, y.hat) mean((y.hat>.5)!=y)
cv.err2<-cv.glm(train,glm.fit,cost = cost)
cv.err2$delta


##### cv for knn
set.seed(1)
klist <- seq(1,21,by=2)
knn<-function(klist,x.train,y.train,x.test){
  n.train<-nrow(x.train)
  n.test<-nrow(x.test)
  p.test<-matrix(NA,n.test,length(klist))
  dsq<-numeric(n.train)
  for(tst in 1:n.test){
    for(trn in 1: n.train){
      dsq[trn]<-sum((x.train[trn,]-x.test[tst,])^2)
    }
    ord<-order(dsq)
    for(ik in 1:length(klist)){
      p.test[tst,ik]<-mean(y.train[ord[1:klist[ik]]])
    }
  }
  invisible(p.test)
}
knn.cv<-function(klist,x.train,y.train,nfolds){
  n.train<-nrow(x.train)
  p.cv<-matrix(NA,n.train,length(klist))
  s<-split(sample(n.train),rep(1:nfolds,length=n.train))
  for(i in seq(nfolds)){
    p.cv[s[[i]],]<-knn(klist,x.train[-s[[i]],],y.train[-s[[i]]],x.train[s[[i]],])
  }
  invisible(p.cv)
}
nfolds<-5
y.pred.train<-(knn(klist,x.train,y.train,x.train))>0.5
y.pred.test<-(knn(klist,x.train,y.train,x.test))>0.5
set.seed(1)
y.pred.cv<-(knn.cv(klist,x.train,y.train,nfolds))>0.5
mis.train<-c()
mis.test<-c()
mis.cv<-c()
for(i in 1:ncol(y.pred.train)){
  mis.train[i]<-mean(y.pred.train[,i] != y.train1)
  mis.test[i]<-mean(y.pred.test[,i] != y.test1)
  mis.cv[i]<-mean(y.pred.cv[,i] != y.train1)
}

plot(mis.train,type = "l",ylim=c(0,0.05),xlab = "k",ylab = "MSE",col=1,lwd=2)
lines(mis.test,col=2,lwd=2)
lines(mis.cv,col=3,lwd=2)
legend("bottomright",legend = c("Train","Test","CV"), lty=1,col = seq(3))

which.min(mis.cv)
mis.test[which.min(mis.cv)]


### cv for ridge
library(glmnet) 
set.seed(1) 
ridge.mod<-glmnet(x.train,y.train,alpha=0)
cv.out<-cv.glmnet(x.train,y.train,alpha=0,nfolds=5)
plot(cv.out)

bestlam<-cv.out$lambda.min
bestlam
ridge.pred<-predict(ridge.mod,s=bestlam,newx=x.test)
mean((ridge.pred-y.test)^2)

ridge.cv<-function(lambda,x.train,y.train,nfolds){
  n.train<-nrow(x.train)
  p.cv<-matrix(NA,n.train,length(lambda))
  s<-split(sample(n.train),rep(1:nfolds,length=n.train))
  for(i in 1:seq(nfolds)){
    ridge.cv<-glmnet(x.train[-s[[i]],],y.train[-s[[i]]],alpha=0)
    p.cv[s[[i]],]<-predict(ridge.cv,s=lambda[i],newx=x.train[s[[i]],])
  }
  invisible(p.cv)
}
set.seed(1)
r.pred.train<-(predict(ridge.mod,s=cv.out$lambda,newx=x.train))>0.5
r.pred.cv <- (ridge.cv(cv.out$lambda,x.train,y.train,nfolds))>0.5
rmis.cv<-c()
for(i in 1:ncol(r.pred.train)){
  rmis.cv[i]<-mean(r.pred.cv[,i] != y.train1)
}
plot(log(cv.out$lambda),rmis.cv)


##### cv for lasso
set.seed(1) 
lasso.mod<-glmnet(x.train,y.train1,alpha=1)
cvl.out<-cv.glmnet(x.train,y.train1,alpha=1,nfolds=5)
plot(cvl.out)

bestlam<-cvl.out$lambda.min
bestlam
lasso.pred<-predict(lasso.mod,s=bestlam,newx=x.test)
mean((lasso.pred-y.test)^2)

set.seed(1)
lasso.cv<-function(lambda,x.train,y.train,nfolds){
  n.train<-nrow(x.train)
  p.cv<-matrix(NA,n.train,length(lambda))
  s<-split(sample(n.train),rep(1:nfolds,length=n.train))
  for(i in 1:seq(nfolds)){
    lasso.cv<-glmnet(x.train[-s[[i]],],y.train[-s[[i]]],alpha=1)
    p.cv[s[[i]],]<-predict(lasso.cv,s=lambda[i],newx=x.train[s[[i]],])
  }
  invisible(p.cv)
}
set.seed(1)
l.pred.train<-(predict(lasso.mod,s=cv.out$lambda,newx=x.train))>0.5
l.pred.cv <- (lasso.cv(cv.out$lambda,x.train,y.train,nfolds))>0.5
lmis.cv<-c()
for(i in 1:ncol(l.pred.train)){
  lmis.cv[i]<-mean(l.pred.cv[,i] != y.train1)
}
plot(log(cv.out$lambda),lmis.cv,ylim = c(min(lmis.cv),max(lmis.cv)))














