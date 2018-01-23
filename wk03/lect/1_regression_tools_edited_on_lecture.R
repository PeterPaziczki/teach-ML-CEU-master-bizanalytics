# install.packages('ggplot2')
# install.packages('lubridate')
# install.packages('ddply')
# install.packages('data.table')
# install.packages('caret')
# install.packages('party')
# install.packages('glmnet')
# install.packages('doParallel')

library(ggplot2)
library(lubridate)
library(plyr)
library(data.table)
library(caret)
library(party)
library(doParallel)


registerDoMC(cores = 4)

GetBikeData <- function(filePath) {
  dt <- fread(filePath)
  dt$quarter    <- factor(dt$season, labels = c("Q1", "Q2", "Q3", "Q4"))
  dt[weather == 4, weather := 3]  # Remove Very Bad weather, since we have only one instance
  dt$weather    <- factor(dt$weather, levels = 1:3, labels = c("Good", "Normal", "Bad"))
  dt$hour       <- hour(ymd_hms(dt$datetime))
  dt$hourFactor <- factor(dt$hour)
  dt$times      <- as.POSIXct(strftime(ymd_hms(dt$datetime), format="%H:%M:%S"), format="%H:%M:%S")
  dt$weekday    <- wday(ymd_hms(dt$datetime))
  return(dt)
}

RootMeanSquaredError <- function(real_value, predicted_value) {
  return(sqrt(mean((real_value - predicted_value) ^ 2)))
}

bikeTrain <- GetBikeData("data/bike_rental/bike_rental_train.csv")
bikeTrain$weather

bikeTest <- GetBikeData("data/bike_rental/bike_rental_test.csv")
bikeTest

ggplot(bikeTrain, aes(x=count)) + geom_histogram() 

##################################################
# 1 Linear Models
##################################################

#   1.a Factor vs integer
##########################

lmSeasonModel <- lm(count~season, bikeTrain)
summary(lmSeasonModel)
postResample(bikeTest$count, predict(lmSeasonModel, bikeTest))
## season is an integer here suggesting that there is a linear relation between count and season, but that is stupid, season should be
## handled as factors, see belowú

# Remember: quarter <- factor(season)
lmQuarterModel <- lm(count~quarter, bikeTrain)
summary(lmQuarterModel)
postResample(bikeTest$count, predict(lmQuarterModel, bikeTest))

# Remember: lmQuarterModel is the same as our ModelByQuarter manual model was (see RMSE)
ModelByQuarter <- function(train, test) {
  new_test = copy(test)
  for (q in unique(train$quarter)) {
    new_test[quarter == q, prediction := mean(train[quarter == q]$count) ]
  }
  return(new_test$prediction)
}
postResample(bikeTest$count, ModelByQuarter(bikeTrain, bikeTest))

## playing with hour as integer
lmHourModel <- lm(count~hour, bikeTrain)
summary(lmHourModel)
postResample(bikeTest$count, predict(lmHourModel, bikeTest))

## playing with hour as factor
lmHourFactorModel <- lm(count~hourFactor, bikeTrain)
summary(lmHourFactorModel)
postResample(bikeTest$count, predict(lmHourFactorModel, bikeTest))

#   1.b Let's build a more complex model
#########################################
lmComplex <- lm(count~quarter+temp+atemp+weather+hour+holiday+workingday+windspeed, bikeTrain)
summary(lmComplex)
postResample(bikeTest$count, predict(lmComplex, bikeTest))

lmHourFactorComplex <- lm(count~quarter+temp+atemp+weather+hourFactor+holiday+workingday+windspeed, bikeTrain)
summary(lmHourFactorComplex)
postResample(bikeTest$count, predict(lmHourFactorComplex, bikeTest))

#   Question: why is lmComplex different from lmHourFactorComplex?

## playing
lmHourFactorComplex <- lm(count~quarter+temp+weather+hourFactor+windspeed+weekday, bikeTrain)
summary(lmHourFactorComplex)
postResample(bikeTest$count, predict(lmHourFactorComplex, bikeTest))

#   1.c Let's introduce caret
#########################################
trctrl <- trainControl(method = "none") # there is no cross validation this time, which we should not do in real life
## we do not use data for validation
lmCaret <- train(count~quarter+temp+atemp+weather+hour+holiday+workingday+windspeed, 
                  data = bikeTrain, 
                  method = "lm",
                  trControl=trctrl
)
lmCaret
postResample(bikeTest$count, predict(lmCaret, bikeTest))

#   1.d Lasso, Ridge and Elastic Net
#########################################

# Remember: Ridge, when alpha to 0. Lasso, when alpha to 1
set.seed(123)
trctrl <- trainControl(method = "cv") # 10-fold cross validation we are doing here
lmElasticNetCaret <- train(count~quarter+temp+atemp+weather+hour+holiday+workingday+windspeed, 
                 data = bikeTrain, 
                 method = "glmnet",
                 trControl=trctrl,
                 tuneLength=5 # it would search for the best five lambda parameters. I can do tuneGrid as well here, see below
                 #tuneGrid = data.frame(lambda=seq(0.1,1,0.1), alpha=rep(1, 10))
) # it was combining  alpha and lambda values looking for the best model
plot(lmElasticNetCaret)
lmElasticNetCaret # the higher the lambda is, the more shrinking we do. It is a grid search. We should give some penalty for too complex models.
## aplha = 1 means we should go with alpha
coef(lmElasticNetCaret$finalModel, lmElasticNetCaret$bestTune$lambda)

summary(lmElasticNetCaret)

postResample(bikeTest$count, predict(lmElasticNetCaret, bikeTest))

# Remember: Interpret the scale and the sign of coefficients

## playing with tuneGrid

set.seed(123)
trctrl <- trainControl(method = "cv")
lmElasticNetCaret <- train(count~quarter+temp+atemp+weather+hour+holiday+workingday+windspeed, 
                           data = bikeTrain, 
                           method = "glmnet",
                           trControl=trctrl,
                           # tuneLength=5
                           tuneGrid = data.frame(lambda=seq(0.11,0.2,0.01), alpha=seq(0.1 , 1, 0.1)) # i have a sequence of aplhas here
) 
plot(lmElasticNetCaret)
lmElasticNetCaret
coef(lmElasticNetCaret$finalModel, lmElasticNetCaret$bestTune$lambda)

summary(lmElasticNetCaret)

postResample(bikeTest$count, predict(lmElasticNetCaret, bikeTest))

## playing again

set.seed(123)
trctrl <- trainControl(method = "cv")
lmElasticNetCaret <- train(count~quarter+temp+atemp+weather+hour+holiday+workingday+windspeed, 
                           data = bikeTrain, 
                           method = "glmnet",
                           trControl=trctrl,
                           # tuneLength=5
                           tuneGrid = data.frame(lambda=seq(100,1000,100), alpha=rep(1, 10)) # i have a sequence of aplhas here
) 
plot(lmElasticNetCaret)
lmElasticNetCaret
coef(lmElasticNetCaret$finalModel, lmElasticNetCaret$bestTune$lambda)

summary(lmElasticNetCaret)

postResample(bikeTest$count, predict(lmElasticNetCaret, bikeTest))


##################################################
# 2 Nearest Neighbors
##################################################


# Exercise 2: 
#   - find additional features (to season, weekday), and find a good combination for best prediction
#   - find a good crossvalidation size (number)
#   - does repeat have an impact?
#   - does preProcessing improve results?
#   - can you improve resulst by setting k values manually?
set.seed(123)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3) # 10-fold cross validation repeated 3 times
knnModel <- train(count~season+workingday, data = bikeTrain, method = "knn",
                 trControl=trctrl,
                 preProcess=c("center", "scale"),
                 tuneLength=10
                 #tuneGrid = data.frame(k=c(2:8))
                 )
knnModel
postResample(bikeTest$count, predict(knnModel, bikeTest))

## playing

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3) # 10-fold cross validation repeated 3 times
knnModel <- train(count~quarter+weekday+temp+hourFactor, data = bikeTrain, method = "knn",
                  trControl=trctrl,
                  preProcess=c("center", "scale"),
                  #tuneLength=10
                  tuneGrid = data.frame(k=c(2:15)) # number of nearest neighbours
)
knnModel
postResample(bikeTest$count, predict(knnModel, bikeTest))


##################################################
# 3 Trees
##################################################

# Exercise 3/A: 
#   - Find good input features 
#   - find maxdepth
#   - find minsplit
library(rpart)

treeSimpleModel <- ctree(count~season+weekday, data=bikeTrain,
                         controls = ctree_control())
RootMeanSquaredError(bikeTest$count, predict(treeSimpleModel, bikeTest))

treeSimpleModel <- ctree(count~season+holiday+workingday+temp+hour, data=bikeTrain,
                         controls = ctree_control())# maxdepth = 10, minsplit=10, mincriterion=0.99
postResample(bikeTest$count, predict(treeSimpleModel, bikeTest))
plot(treeSimpleModel)

# Exercise 3/B:
#   - cp (minimum R^2 improvement) parameter is new here, can you find the best values for it?
#   - would rpart2 (using maxdepth) improve results? 
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
treeCPModel <- train(count~quarter+holiday+workingday+temp+hour, data = bikeTrain, method = "rpart",
                  trControl=trctrl,
                  tuneLength = 20
                  #tuneGrid = data.frame(cp=seq(0.000001, 0.001, 0.0001))
                  )
treeCPModel
postResample(bikeTest$count, predict(treeCPModel, bikeTest))

treeMDModel <- train(count~quarter+holiday+workingday+temp+hour, data = bikeTrain, method = "rpart2",
                     trControl=trctrl,
                     tuneLength = 20
                     #tuneGrid = data.frame(maxdepth=seq(1,10))
)
treeMDModel
postResample(bikeTest$count, predict(treeMDModel, bikeTest))


##################################################
# 4 Support Vector Machine
##################################################

#   4.a Linear model
#################################################
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(123)
svmLinearModel <- train(count~quarter+holiday+workingday+temp+hour, 
                    data = bikeTrain, 
                    method = "svmLinear",
                    trControl=trctrl,
                    #preProcess = c("center", "scale"),
                    tuneLength = 10)
plot(svmLinearModel)
postResample(bikeTest$count, predict(svmLinearModel, bikeTest))

#   4.b Non-Linear model
#################################################

set.seed(123)
svmRadialModel <- train(count~quarter+temp+hour, 
                        data = bikeTrain, 
                        method = "svmRadial",
                        trControl=trctrl,
                        tuneLength = 10)
svmRadialModel
plot(svmRadialModel)
postResample(bikeTest$count, predict(svmRadialModel, bikeTest))  # 129.02

