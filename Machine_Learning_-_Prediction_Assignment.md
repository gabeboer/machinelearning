---
title: "Machine Learning - Prediction"
author: "Gabe Boer"
date: "13 november 2017"
output: 
  html_document: 
    fig_height: 4
    keep_md: yes
---



## Excecutive Summary
This document shows what machine learning can do in predicting whether the barbell lift was correctly or not (5 different classes). A number of machine learning models have been applied and cross validated. The random forest model has the highest accuracy.

## Data

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Getting the data
Getting the data and store in local variables 'training' and 'test', using read.csv2 function.


```r
urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(urlTrain, destfile = "pml-training.csv")
download.file(urlTest, destfile = "pml-testing.csv")
training <- read.csv2("pml-training.csv", sep=",", dec = ".")
test <- read.csv2("pml-testing.csv", sep=",", dec = ".")

#View(test)
dim(training)
```

```
## [1] 19622   160
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

### Libraries and seed

```
## Warning: package 'caret' was built under R version 3.4.2
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'klaR' was built under R version 3.4.2
```

```
## Loading required package: MASS
```

```
## Warning: package 'lubridate' was built under R version 3.4.2
```

```
## 
## Attaching package: 'lubridate'
```

```
## The following object is masked from 'package:plyr':
## 
##     here
```

```
## The following object is masked from 'package:base':
## 
##     date
```

```
## Warning: package 'dplyr' was built under R version 3.4.2
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:lubridate':
## 
##     intersect, setdiff, union
```

```
## The following object is masked from 'package:MASS':
## 
##     select
```

```
## The following objects are masked from 'package:plyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```
## Warning: package 'randomForest' was built under R version 3.4.2
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## Warning: package 'rpart.plot' was built under R version 3.4.2
```
## Data Preparation

### Removing the 'near zero' dimensions
160 columns is a lot. In this step we'll reduce the number of dimensions. 
The first 7 columns will not help in prediction and will be removed (rec.no., username, time related). Next: every variable with over 90% NA's can be removed.
A number of variables will have a high correlation (with PCA dimension reduction)


```r
#replace the DIV/0 by NA
#revalue(training$kurtosis_yaw_belt, c("#DIV/0!"=NA))
#revalue(training$kurtosis_yaw_forearm, c("#DIV/0!"=NA))
training[training=="#DIV/0!"] <- NA
NearZero <- nearZeroVar(training, saveMetrics = TRUE)
training=training[,NearZero$nzv==FALSE]

NAcount = training %>% summarize_all(funs(sum(is.na(.)) / length(.)))
NAcol = which(NAcount>0.9)
colnames(training[,c(1:7)])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"          
## [7] "roll_belt"
```

```r
training=training[,-NAcol]
training=training[,-c(1:7)]
#and make sure test equals training setup
test=test[,NearZero$nzv==FALSE]
test=test[,-NAcol]
test=test[,-c(1:7)]
```

### Splitting the training set
In the end we need to have to use the 20 records as available in the test dataset.
To test my model, I'll split the training set in a training and a validation set (70/30).

```r
inTrain <- createDataPartition(y=training$classe, p=0.7, list = FALSE)
train7 <- training[inTrain,]
train3 <- training[-inTrain,]
```

## The models

A number of models is trained on the 70% of the training subset and then validated against the train3 set.

### Decision tree

```r
dt <- train(classe~., data=train7, method="rpart")
pdt <- predict(dt, train3)     
confmatDT <- confusionMatrix(pdt, train3$classe)
confmatDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1504  473  475  446  219
##          B   45  401   55  177  246
##          C   83  124  423  138  152
##          D   31  140   73  203  159
##          E   11    1    0    0  306
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4821          
##                  95% CI : (0.4692, 0.4949)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3229          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8984  0.35206  0.41228  0.21058  0.28281
## Specificity            0.6170  0.88980  0.89772  0.91811  0.99750
## Pos Pred Value         0.4825  0.43398  0.45978  0.33498  0.96226
## Neg Pred Value         0.9386  0.85124  0.87855  0.85584  0.86061
## Prevalence             0.2845  0.19354  0.17434  0.16381  0.18386
## Detection Rate         0.2556  0.06814  0.07188  0.03449  0.05200
## Detection Prevalence   0.5297  0.15701  0.15633  0.10297  0.05404
## Balanced Accuracy      0.7577  0.62093  0.65500  0.56434  0.64016
```
### Random Forest

```r
rf <- train(classe~., data=train7, method="rf", prox=TRUE)
prf <- predict(rf, train3)
confmatRF <- confusionMatrix(prf, train3$classe)
confmatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674   12    0    0    0
##          B    0 1119   10    0    0
##          C    0    7 1014   15    3
##          D    0    0    2  946    0
##          E    0    1    0    3 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.991           
##                  95% CI : (0.9882, 0.9932)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9886          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9824   0.9883   0.9813   0.9972
## Specificity            0.9972   0.9979   0.9949   0.9996   0.9992
## Pos Pred Value         0.9929   0.9911   0.9759   0.9979   0.9963
## Neg Pred Value         1.0000   0.9958   0.9975   0.9964   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1901   0.1723   0.1607   0.1833
## Detection Prevalence   0.2865   0.1918   0.1766   0.1611   0.1840
## Balanced Accuracy      0.9986   0.9902   0.9916   0.9905   0.9982
```
### other models (appendix)
The NB and LDA models didn't score high in accuracy and will be ignored. 

## Scoring the test set
In order to answer the quiz questions, the random forest model is used to predict the 20 rows of the test dataset.


```r
quiz <- predict(rf, test)
quiz
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion
The random forest model performs superior to the other model.

## Appendix

Find some additional info on models here.
models:

nb naive bayes (independence between variables, hmm....), not included in KNITR due to time out error
lda linear discriminant analysis (idem)


```r
#nb <- train(classe~., data=train7, method="nb")
#lda <- train(classe~., data=train7, method="lda")
#pnb <- predict(nb, train3)
#plda <- predict(lda, train3)
#table(plda, pnb)
#confmatNB <- confusionMatrix(pnb, train3$classe)
#confmatNB
#confmatLDA <- confusionMatrix(plda, train3$classe)
#confmatLDA
```




