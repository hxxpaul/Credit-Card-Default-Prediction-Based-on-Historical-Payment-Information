library(MASS)
library(caret)
library(gmodels)
library(pROC)
library(e1071)
library(neuralnet)
library(corrplot)
d <- gdata::read.xls('./data/default of credit card clients.xls')

# data cleaning and preparation
# convert categorical variables into factors
d$Default.Payment.Next.Month <- factor(d$Default.Payment.Next.Month)
# detect missing values
isbad.edu <- which(d$EDUCATION == 0)
isbad.mar <- which(d$MARRIAGE == 0)
d$EDUCATION[isbad.edu] = NA
d$MARRIAGE[isbad.mar] = NA
sum(complete.cases(d))/nrow(d) # only 0.2% of incidents have missing values, so it's safe to drop them
d <- na.omit(d)
# standardization
d.s_1 <- as.data.frame(scale(d[, 1:23]))
d.s_2 <- dplyr::select(d, 24)
d.s <- cbind(d.s_1, d.s_2)
# identify and discard the outliers
source("http://goo.gl/UUyEzD")
outlierKD(d.s, LIMIT_BAL)
outlierKD(d.s, AGE)
outlierKD(d.s, BILL_AMT1)
outlierKD(d.s, BILL_AMT2)
outlierKD(d.s, BILL_AMT3)
outlierKD(d.s, BILL_AMT4)
outlierKD(d.s, BILL_AMT5)
outlierKD(d.s, BILL_AMT6)
outlierKD(d.s, PAY_AMT1)
outlierKD(d.s, PAY_AMT2)
outlierKD(d.s, PAY_AMT3)
outlierKD(d.s, PAY_AMT4)
outlierKD(d.s, PAY_AMT5)
outlierKD(d.s, PAY_AMT6)
d.s <- na.omit(d.s)
# split dataset into training and testing sets
trainsize <- round(nrow(d.s)*0.95)
testsize <- nrow(d.s) - trainsize
set.seed(666)
training_indices <- sample(seq_len(nrow(d.s)), size = trainsize)
trainSet <- d.s[training_indices, ]
testSet <- d.s[-training_indices, ]
# balance response variable in training and testing set to 50:50
trainSet_0 <- dplyr::filter(trainSet, Default.Payment.Next.Month == 0)
trainSet_1 <- dplyr::filter(trainSet, Default.Payment.Next.Month == 1)
trainSet_0 <- trainSet_0[sample(nrow(trainSet_0), 4762), ]
trainSet <- rbind(trainSet_0, trainSet_1)

# analysis
# linear discriminant analysis
lda <- lda(Default.Payment.Next.Month ~ ., data = trainSet)
lda.pre <- predict(lda, newdata = testSet)
CrossTable(x = testSet$Default.Payment.Next.Month, y = lda.pre$class, prop.chisq = FALSE) # 68.4%

# knn
trctrl <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
set.seed(123)
knn <- train(Default.Payment.Next.Month ~ ., data = trainSet, method = 'knn',
                 trControl = trctrl, preProcess = c('center'), tuneLength = 10)
knn
knn.pre <- predict(knn, newdata = testSet)
CrossTable(x = testSet$Default.Payment.Next.Month, y = knn.pre, prop.chisq = FALSE) # 72.2%

# logistic regression
logit <- glm(Default.Payment.Next.Month ~ ., data = trainSet, family = 'binomial')
summary(logit)
logit.pre <- predict(logit, newdata = testSet, type = 'response')
CrossTable(x = testSet$Default.Payment.Next.Month, y = logit.pre>0.5, prop.chisq = FALSE) # 68.7%
# ROC curve
plot(roc(testSet$Default.Payment.Next.Month, logit.pre, direction = '<'), col = 'red', lwd = 3)

# naive bayes
nb <- naiveBayes(Default.Payment.Next.Month ~ ., data = trainSet)
nb.pre <- predict(nb, newdata = testSet)
CrossTable(x = testSet$Default.Payment.Next.Month, y = nb.pre, prop.chisq = FALSE) # 74.9%

# neural network
set.seed(321)
mm <- model.matrix(~ Default.Payment.Next.Month + SEX + EDUCATION + MARRIAGE + PAY_1 + PAY_2 + PAY_3 + 
                     PAY_4 + PAY_5 + PAY_6 + AGE + LIMIT_BAL + BILL_AMT1 + BILL_AMT2 +BILL_AMT3 + 
                     BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + 
                     PAY_AMT5 + PAY_AMT6, data = trainSet)
nn <- neuralnet(Default.Payment.Next.Month1 ~ SEX + EDUCATION + MARRIAGE + PAY_1 + PAY_2 + PAY_3 + 
                  PAY_4 + PAY_5 + PAY_6 + AGE + LIMIT_BAL + BILL_AMT1 + BILL_AMT2 +BILL_AMT3 + 
                  BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + 
                  PAY_AMT5 + PAY_AMT6, mm, hidden = 3, linear.output = TRUE)
plot(nn)
nn.pre <- compute(nn, testSet[1:23])
nn.pre$net.result <- sapply(nn.pre$net.result, round, digit = 0)
CrossTable(x = testSet$Default.Payment.Next.Month, y = nn.pre$net.result>0.5, prop.chisq = FALSE) # 72.7%

# ensemble
# majority voting
rt.lda <- lda.pre$class
rt.knn <- knn.pre
rt.logit <- ifelse(logit.pre > 0.5, 1, 0)
rt.nb <- nb.pre
rt.nn <- nn.pre$net.result
maj.pre <- as.factor(ifelse(rt.lda==1 & rt.knn==1 & rt.logit==1, 1, 
                            ifelse(rt.lda==1 & rt.logit==1 & rt.nb==1, 1, 
                            ifelse(rt.lda==1 & rt.logit==1 & rt.nn==1, 1,
                            ifelse(rt.lda==1 & rt.nb==1 & rt.nn==1, 1,
                            ifelse(rt.lda==1 & rt.knn==1 & rt.nb==1, 1,
                            ifelse(rt.lda==1 & rt.knn==1 & rt.nn==1, 1,
                            ifelse(rt.knn==1 & rt.logit==1 & rt.nb==1, 1,
                            ifelse(rt.knn==1 & rt.logit==1 & rt.nn==1, 1,
                            ifelse(rt.knn==1 & rt.nb==1 & rt.nn==1, 1,
                            ifelse(rt.logit==1 & rt.nb==1 & rt.nn==1, 1, 0)))))))))))
CrossTable(x = testSet$Default.Payment.Next.Month, y = maj.pre, prop.chisq = FALSE) # 72.9%

# pca
# correlation visualization
cor <- round(cor(d.s[, 1:23]), digits = 3)
corrplot(cor, type = 'upper', order = 'hclust', tl.col = 'black', tl.srt = 45, tl.cex = 0.7)
# pca computing and visualization
pca <- prcomp(d.s[, 1:23], center = TRUE, scale. = TRUE)
summary(pca)
plot(pca, type = 'l')
ev <- pca$sdev^2
evplot <- function(ev)
{
  # Broken stick model (MacArthur 1957)
  n <- length(ev)
  bsm <- data.frame(j=seq(1:n), p=0)
  bsm$p[1] <- 1/n
  for (i in 2:n) bsm$p[i] <- bsm$p[i-1] + (1/(n + 1 - i))
  bsm$p <- 100*bsm$p/n
  # Plot eigenvalues and % of variation for each axis
  op <- par(mfrow=c(2,1))
  barplot(ev, main="Eigenvalues", col="bisque", las=2)
  abline(h=mean(ev), col="red")
  legend("topright", "Average eigenvalue", lwd=1, col=2, bty="n")
  barplot(t(cbind(100*ev/sum(ev), bsm$p[n:1])), beside=TRUE, 
          main="% variation", col=c("bisque",2), las=2)
  legend("topright", c("% eigenvalue", "Broken stick model"), 
         pch=15, col=c("bisque",2), bty="n")
  par(op)
}
evplot(ev) # according to Kaiser's Rule, the first six components should be chosen
pca.c <- pca$x[ , 1:6]
pca.c <- cbind(pca.c, dplyr::select(d.s, Default.Payment.Next.Month))

# split dataset into training and testing sets
trainsize.pca <- round(nrow(pca.c)*0.95)
testsize.pca <- nrow(pca.c) - trainsize.pca
set.seed(666)
training_indices.pca <- sample(seq_len(nrow(pca.c)), size = trainsize.pca)
trainSet.pca <- pca.c[training_indices.pca, ]
testSet.pca <- pca.c[-training_indices.pca, ]
# balance response variable in training and testing set to 50:50
trainSet_0.pca <- dplyr::filter(trainSet.pca, Default.Payment.Next.Month == 0)
trainSet_1.pca <- dplyr::filter(trainSet.pca, Default.Payment.Next.Month == 1)
trainSet_0.pca <- trainSet_0.pca[sample(nrow(trainSet_0.pca), 4762), ]
trainSet.pca <- rbind(trainSet_0.pca, trainSet_1.pca)

# pca.lda
lda.pca <- lda(Default.Payment.Next.Month ~ ., data = trainSet.pca)
lda.pre.pca <- predict(lda.pca, newdata = testSet.pca)
CrossTable(x = testSet.pca$Default.Payment.Next.Month, y = lda.pre.pca$class, prop.chisq = FALSE) # 68.3%

# knn
trctrl.pca <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
set.seed(123)
knn.pca <- train(Default.Payment.Next.Month ~ ., data = trainSet.pca, method = 'knn',
             trControl = trctrl.pca, preProcess = c('center'), tuneLength = 10)
knn.pca
knn.pre.pca <- predict(knn.pca, newdata = testSet.pca)
CrossTable(x = testSet.pca$Default.Payment.Next.Month, y = knn.pre.pca, prop.chisq = FALSE) # 69.4%

# logistic regression
logit.pca <- glm(Default.Payment.Next.Month ~ ., data = trainSet.pca, family = 'binomial')
summary(logit.pca)
logit.pre.pca <- predict(logit.pca, newdata = testSet.pca, type = 'response')
CrossTable(x = testSet.pca$Default.Payment.Next.Month, y = logit.pre.pca>0.5, prop.chisq = FALSE) # 68.3%

# naive bayes
nb.pca <- naiveBayes(Default.Payment.Next.Month ~ ., data = trainSet.pca)
nb.pre.pca <- predict(nb.pca, newdata = testSet.pca)
CrossTable(x = testSet.pca$Default.Payment.Next.Month, y = nb.pre.pca, prop.chisq = FALSE) # 71%

# neural network
set.seed(321)
mm.pca <- model.matrix(~ Default.Payment.Next.Month + PC1 + PC2 + PC3 + PC4 + PC5 + PC6, data = trainSet.pca)
nn.pca <- neuralnet(Default.Payment.Next.Month1 ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6, mm.pca, hidden = 3, linear.output = TRUE)
plot(nn.pca)
nn.pre.pca <- compute(nn.pca, testSet.pca[1:6])
nn.pre.pca$net.result <- sapply(nn.pre.pca$net.result, round, digit = 0)
CrossTable(x = testSet.pca$Default.Payment.Next.Month, y = nn.pre.pca$net.result>0.5, prop.chisq = FALSE) # 73.9%
