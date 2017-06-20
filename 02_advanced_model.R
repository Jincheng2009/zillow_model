library(ggplot2)
library(data.table)
library(plyr)
library(MASS)
library(Matrix)
library(xgboost)
rm(list=ls())
########################################################################
####################### Functions ######################################
########################################################################
## Check the type and NA value fraction of each column in the data frame
checkType <- function(df) {
    type.df <- data.frame(freq=sapply(df, function(x) round(sum(is.na(x)) / length(x) * 100, 3)), 
                          type=sapply(df, class))
    type.df$name <- rownames(type.df)
    rownames(type.df) <- NULL
    type.df <- type.df[,c("name", "type", "freq")]
    type.df$nlevel <- sapply(df, function(x) {
        if(class(x) == "factor") {
            length(unique(x))
        } else {
            0
        }
    })
    type.df
}

########################################################################
########################### Data import ################################
########################################################################
datapath <- "D:/data_science/zillow/data"
df.x <- suppressWarnings(fread(paste(datapath, "properties_2016.csv", sep="/")))
df.y <- fread(paste(datapath, "train_2016_v2.csv", sep="/"))
df.y.test <- fread(paste(datapath, "lm_initial.csv", sep="/"), header = TRUE)

df.y.test <- data.frame(df.y.test, check.names = FALSE)
df.x <- data.frame(df.x)
########################################################################
###################### Datatype conversion #############################
########################################################################
## Boolean data conversion
df.x$hashottuborspa <- ifelse(df.x$hashottuborspa=="true", TRUE, FALSE)
df.x$pooltypeid10 <- ifelse(df.x$pooltypeid10==1, TRUE, FALSE)
df.x$pooltypeid2 <- ifelse(df.x$pooltypeid2==1, TRUE, FALSE)
df.x$pooltypeid7 <- ifelse(df.x$pooltypeid7==1, TRUE, FALSE)
df.x$fireplaceflag <- ifelse(df.x$fireplaceflag=="true", TRUE, FALSE)

df.x$taxdelinquencyyear[is.na(df.x$taxdelinquencyyear)] <- 16
df.x$taxdelinquencyyear <- as.Date(as.character(df.x$taxdelinquencyyear), "%y")
df.x$taxdelinquencyyear <- as.numeric(format(df.x$taxdelinquencyyear, "%Y"))

df.x$taxdelinquencytype <- 0
df.x$taxdelinquencytype[df.x$taxdelinquencyyear==2015] <- 1
df.x$taxdelinquencytype[df.x$taxdelinquencyyear %in% c(2013, 2014)] <- 2
df.x$taxdelinquencytype[df.x$taxdelinquencyyear < 2013] <- 3
df.x$taxdelinquencytype <- as.factor(df.x$taxdelinquencytype)

## Fill in fireplace cnt
df.x$fireplacecnt[is.na(df.x$fireplacecnt)] <- 0
df.x$poolcnt[is.na(df.x$poolcnt)] <- 0
df.x$threequarterbathnbr[is.na(df.x$threequarterbathnbr)] <- 0

## Merge pooltypeid2/7/10 into one variable
df.x$pooltypeid <- 0
df.x[which(df.x$pooltypeid2),]$pooltypeid <- 2
df.x[which(df.x$pooltypeid7),]$pooltypeid <- 7
df.x[which(df.x$pooltypeid10),]$pooltypeid <- 10

factor.vars <- c("airconditioningtypeid", 
                 "architecturalstyletypeid",
                 "decktypeid", 
                 "heatingorsystemtypeid",
                 "propertycountylandusecode", 
                 "propertylandusetypeid", 
                 "propertyzoningdesc",
                 #"regionidcity", 
                 "regionidcounty", 
                 "regionidneighborhood", 
                 #"regionidzip",
                 "storytypeid", 
                 "fips",
                 "typeconstructiontypeid", 
                 # "rawcensustractandblock", 
                 # "censustractandblock", 
                 "buildingclasstypeid", 
                 "pooltypeid")
for(v in factor.vars) {
    df.x[,v] <- as.factor(df.x[,v])
}

for(v in colnames(df.x)) {
    if(class(df.x[,v]) %in% c("integer", "integer64")) {
        df.x[,v] <- as.numeric(df.x[,v])
    } else if(class(df.x[,v]) %in% c("logical", "character")) {
        df.x[,v] <- as.factor(df.x[,v])
    }
}

########################################################################
########## Remove cols that are redundant (1st pass) ###################
########################################################################
## variables to be removed for model training
remove.vars <- c("pooltypeid2",    # Encoded in pooltypeid
                 "pooltypeid7",    # Encoded in pooltypeid
                 "pooltypeid10"   # Encoded in pooltypeid
                 )

remove.idx <- which(colnames(df.x) %in% remove.vars)

df.x <- df.x[,-remove.idx]

########################################################################
################# Remove rows that are mostly NAs ######################
########################################################################
#row.keep <- which(apply(df.x,1,function(x) sum(is.na(x))) < 30)
#df.x <- df.x[row.keep,]

log.vars <- c("calculatedfinishedsquarefeet","finishedsquarefeet12",
              "finishedsquarefeet15",
              "finishedsquarefeet50", # log10(x+500)
              "lotsizesquarefeet",
              "yardbuildingsqft17",
              "structuretaxvaluedollarcnt",
              "taxvaluedollarcnt",
              "landtaxvaluedollarcnt",
              "taxamount")

for(v in log.vars) {
    df.x[,v] <- log10(df.x[,v])
}

data <- merge(df.y, df.x)
data <- data.frame(data)
data <- subset(data, abs(logerror) < 1.)
###################### Create month variable ###########################
data$date <- strptime(data$transactiondate, format="%Y-%m-%d")
data$month <- format(data$date, "%m")
data$month <- as.factor(data$month)
remove.vars <- c("date", 
                 "transactiondate")
remove.idx <- which(colnames(data) %in% remove.vars)
data <- data[,-remove.idx]

######################################
data1 <- subset(data, !is.na(calculatedbathnbr) & 
                    !is.na(calculatedfinishedsquarefeet) &
                    !is.na(fullbathcnt) &
                    !is.na(regionidzip) &
                    !is.na(regionidcity) &
                    !is.na(yearbuilt) &
                    !is.na(taxvaluedollarcnt) &
                    !is.na(taxamount) &
                    !is.na(lotsizesquarefeet))

type.df <- checkType(data1)

########################################################################
######################## Boosting Tree xgboost #########################
########################################################################
## Exclude logerror and parcelid
type.df <- checkType(data1)
required.vars <- subset(type.df, !name %in% c("parcelid", 
                                         "logerror", 
                                         "taxdelinquencyyear", 
                                         "taxdelinquencyflag",
                                          "propertyzoningdesc"
                                         ) &
                                         freq==0)$name
all.vars <- required.vars
#all.vars <- c(required.vars, 
#              "unitcnt")
# "propertyzoningdesc"
# "taxdelinquencyyear"
# "taxdelinquencyflag"
set.seed(12)
train.idx <- sample(nrow(data1), 0.7*nrow(data1))

formula <- as.formula(paste("logerror", paste(all.vars, collapse=" + "), sep=" ~ "))

## Add more variables
options(na.action='na.pass')

data.x <- sparse.model.matrix(formula, data=data1)
dall <- xgb.DMatrix(data = data.x, label = data1$logerror)
dtrain <- xgb.DMatrix(data = data.x[train.idx,], label = data1$logerror[train.idx])
dtest <- xgb.DMatrix(data = data.x[-train.idx,], label = data1$logerror[-train.idx])

# Cross validation to find the optimal parameters
bst.cv <- xgb.cv(data = dall, nrounds = 1000, maxdepth = 4, base_score = 0,
                 params = list(eta=0.02), eval_metric = "mae", min_child_weight = 0.05,
                 nfold=6, colsample_bytree = 0.5, 
                 early_stopping_rounds = 10)

fit <- xgboost(data = dtrain, nrounds = 58, maxdepth = 4, base_score = 0,
               params = list(eta=0.02), eval_metric = "mae", min_child_weight = 0.05,
               early_stopping_rounds = 10, colsample_bytree = 0.5)

## Fit all data to improve accuracy
fit <- xgboost(data = dall, nrounds = 60, maxdepth = 4, base_score = 0,
               params = list(eta=0.02), eval_metric = "mae", min_child_weight = 0.05,
               early_stopping_rounds = 10, colsample_bytree = 0.5)
########################################################################
######################### Model evaluation #############################
########################################################################
train.pred <- predict(fit, dtrain)
train.y <- data1[train.idx,]$logerror
train.mae <- mean(abs(train.pred - train.y))
plot(train.pred, train.y)

test.pred <- predict(fit, dtest)
test.y <- data1[-train.idx,]$logerror
test.mae <- mean(abs(test.pred - test.y))
plot(test.pred, test.y)

paste("training MAE: ", round(train.mae,6))
paste("testing MAE: ", round(test.mae,6))
paste("benchmarking testing MAE: ", round(mean(abs(test.y)), 6))
paste("Improvement: " , format(mean(abs(test.y)) - test.mae, scientific=TRUE, digits=3))

#######################################################################
## Predict all the qualified data
#######################################################################
## Select the qualified data
## Month variable generated later
required.vars <- required.vars[required.vars!="month"]
all.vars <- all.vars[all.vars!="month"]
df.x1 <- df.x[,required.vars]
row.keep <- apply(df.x1, 1, function(x) sum(is.na(x)) < 1)
df.x1 <- df.x[row.keep,all.vars]
parcelid <- df.x$parcelid[row.keep]

## Prepare data for xgboost prediction
prepareData <- function(df.temp) {
    formula <- as.formula(paste("", paste(all.vars, collapse=" + "), sep=" ~ "))
    data.x1 <- sparse.model.matrix(formula, data=df.temp)
    dmat <- xgb.DMatrix(data = data.x1)
    dmat
}
# October
df.x1$month <- factor("10", levels=c("01","02","03","04","05","06","07","08","09","10","11","12"))
dmat <- prepareData(df.x1)
df.y.pred <- predict(fit, dmat)
# November
df.x1$month <- factor("11", levels=c("01","02","03","04","05","06","07","08","09","10","11","12"))
dmat <- prepareData(df.x1)
df.y.pred <- cbind(df.y.pred, predict(fit, dmat))
# December
df.x1$month <- factor("12", levels=c("01","02","03","04","05","06","07","08","09","10","11","12"))
dmat <- prepareData(df.x1)
df.y.pred <- cbind(df.y.pred, predict(fit, dmat))
# Copy for second year
df.y <- cbind(df.y.pred, df.y.pred)


df.y <- data.frame(round(df.y, 6))
df.y <- cbind(parcelid, df.y)
colnames(df.y) <- colnames(df.y.test)
df.y$ParcelId <- as.character(df.y$ParcelId)

## Keep the predictions that current model could not handle
df.y.missing <- subset(df.y.test, ! ParcelId %in% df.y$ParcelId)

df.y.final <- rbind(df.y, df.y.missing)
df.y.final <- df.y.final[match(df.y.test$ParcelId, df.y.final$ParcelId),]

## Linear regression model with fips, month and area as predictors
write.csv(df.y.final, file=paste(datapath, "../results/xgboost_4.csv", sep="/"), row.names = FALSE, quote=FALSE)

## Investigate feature importance for model improvement
importance_matrix <- xgb.importance(colnames(data.x), model = fit)
pdf(paste(datapath, "feature_importance_xgboost.pdf", sep="/"), height =20)
xgb.plot.importance(importance_matrix[1:50,], rel_to_first = TRUE, xlab = "Relative importance")
dev.off()

tax.error <- ddply(data1, .(taxdelinquencyyear), summarize, mean(logerror))
