library(ggplot2)
library(data.table)
library(plyr)

########################################################################
########################### Data import ################################
########################################################################
datapath <- "D:/data_science/zillow/data"
## Use data.table to speed up data import
df.x <- suppressWarnings(fread(paste(datapath, "properties_2016.csv", sep="/")))
df.y <- fread(paste(datapath, "train_2016_v2.csv", sep="/"))
df.y.test <- fread(paste(datapath, "sample_submission.csv", sep="/"), header=TRUE)
df.x <- data.frame(df.x)
df.y.test <- data.frame(df.y.test, check.names=FALSE)

########################################################################
############### Prediction of data with month ##########################
########################################################################
data <- merge(df.x, df.y)
## remove outlier
data1 <- subset(data, abs(logerror) < 1.)

###################### Create month variable ###########################
data1$date <- strptime(data1$transactiondate, format="%Y-%m-%d")
data1$month <- format(data1$date, "%m")
data1$month <- as.factor(data1$month)
remove.vars <- c("date", 
                 "transactiondate")
remove.idx <- which(colnames(data1) %in% remove.vars)
data1 <- data1[,-remove.idx]

## fit model
fit <- lm(logerror ~ month, data = data1)

temp.x <- data.frame(month=factor(c("10", "11", "12"), 
                                  levels=c("01","02","03","04","05","06","07","08","09","10","11","12")))
temp.y <- predict(fit, temp.x)

df.y.test$`201610` <- round(temp.y[1],6)
df.y.test$`201710` <- round(temp.y[1],6)
df.y.test$`201611` <- round(temp.y[2],6)
df.y.test$`201711` <- round(temp.y[2],6)
df.y.test$`201612` <- round(temp.y[3],6)
df.y.test$`201712` <- round(temp.y[3],6)
write.csv(df.y.test, file=paste(datapath, "lm_initial.csv", sep=""), row.names = FALSE, quote=FALSE)

