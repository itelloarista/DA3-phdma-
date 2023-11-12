#Irene Tello Arista: Prediction with Machine Learning for Economists 2023/24 Fall
#Predicting earnings per hour according to the performance of 4 models based on
#RMSE, cross-validation RMSE and BIC

#The objective of this script is to identify which of the following models works 
#better for predicting earnings per hour of an occupation according to the 
#following variables: gender, age, education, and hours worked

#The data used is the cps-earnings dataset based on the Current Population Survey

#Clear memory 
rm(list=ls())

#Packages used

#install.packages('lspline')
#install.packages('cowplot')
#install.packages('tidyverse')
#install.packages('boot')
#install.packages('estimatr')
#install.packages('huxtable')
#install.packages('stargazer')
#install.packages('modelsummary')
#install.packages('urca')
#install.packages('plm')


library(tidyverse)
library(lspline)
library(cowplot)
library(boot)
library(estimatr)
library(huxtable)
library(stargazer)
library(modelsummary)
library(urca)
library(fixest)
library(lmtest)
library(sandwich)
library(caret)

#Set working directory 

setwd("/Users/irenetelloarista/Desktop/da_data_repo/cps-earnings/clean")

#Upload data
cps <- read_csv('/Users/irenetelloarista/Desktop/da_data_repo/cps-earnings/clean/morg-2014-emp.csv')

head(cps)
colnames(cps)


#SELECT OCCUPATION
# keep only 1 occupation types: Psychologists 1820

cps <- subset(cps, occ2012 == 1820)
head(cps)

#Select variables 
cps <- cps %>% mutate(female=sex==2,
                      w=earnwke/uhours,
                      lnw=log(w)
)

# EDUC
cps <- cps %>% mutate(ed_MA = as.numeric(grade92==44),
                      ed_Profess = as.numeric(grade92==45),
                      ed_PhD = as.numeric(grade92==46)
)

cps <- cps %>% mutate(white=as.numeric(race==1),
                      afram = as.numeric(race==2),
                      asian = as.numeric(race==4),
                      hisp = !is.na(ethnic),
                      othernonw = as.numeric(white==0 & afram==0 & asian==0 & hisp==0),
                      nonUSborn = as.numeric(prcitshp=="Foreign Born, US Cit By Naturalization" | prcitshp=="Foreign Born, Not a US Citizen") 
)


# Potentially endogeneous demographics
cps <- cps %>% mutate(married = as.numeric(marital==1 | marital==2),
                      divorced = as.numeric(marital==3 | marital==5 | marital==6),
                      wirowed = as.numeric(marital==4),
                      nevermar = as.numeric(marital==7),
                      
                      child0 = as.numeric(chldpres==0),
                      child1 = as.numeric(chldpres==1),
                      child2 = as.numeric(chldpres==2),
                      child3 = as.numeric(chldpres==3),
                      child4pl = as.numeric(chldpres>=4))

# Work-related variables
cps <- cps %>% mutate(fedgov = as.numeric(class=="Government - Federal"),
                      stagov = as.numeric(class=="Government - State"),
                      locgov = as.numeric(class=="Government - Local"),
                      nonprof = as.numeric(class=="Private, Nonprofit"),
                      ind2dig = as.integer(as.numeric(as.factor(ind02))/100),
                      occ2dig = as.integer(occ2012/100),
                      union = as.numeric(unionmme=="Yes" | unioncov=="Yes"))


###################################
# Linear regressions

# Model 1: Linear regression on age
model1 <- as.formula(lnw ~ age)
# Models 2-5: Multiple linear regressions
model2 <- as.formula(lnw ~ age + female)
model3 <- as.formula(lnw ~ age + female + ed_MA + ed_PhD + ed_Profess)
model4 <- as.formula(lnw ~ age + female + ed_MA + ed_PhD + ed_Profess + white + afram + asian)

# Running simple OLS
reg1 <- feols(model1, data=cps, vcov = 'hetero')
reg2 <- feols(model2, data=cps, vcov = 'hetero')
reg3 <- feols(model3, data=cps, vcov = 'hetero')
reg4 <- feols(model4, data=cps, vcov = 'hetero')

# evaluation of the models
fitstat_register("k", function(x){length( x$coefficients ) - 1}, "No. Variables")
etable( reg1 , reg2 , reg3 , reg4, fitstat = c('aic','bic','rmse','r2','n','k') )

models <- c("reg1", "reg2","reg3", "reg4")
AIC <- c()
BIC <- c()
RMSE <- c()
RSquared <- c()
regr <- c()
k <- c()

# Get for all models (this part of the code is not working)
for ( i in 1:length(models)){
  AIC[i] <- AIC(get(models[i]))
  BIC[i] <- BIC(get(models[i]))
  RMSE[i] <- sqrt(mean((residuals(get(models[i])))^2))
  RSquared[i] <-summary(get(models[i]))$r.squared
  regr[[i]] <- coeftest(get(models[i]), vcov = sandwich)
  k[i] <- get(models[i])$rank -1
}
#Check if something is missing 

summary(reg1)
summary(reg2)
summary(reg3)
summary(reg4)

# All models
eval <- data.frame(models, k, RSquared, BIC)
eval <- eval[complete.cases(eval), ]

eval <- data.frame(models, k, RSquared, RMSE, BIC)
eval <- eval %>%
  mutate(models = paste0("(",gsub("reg","",models),")")) %>%
  rename(Model = models, "R-squared" = RSquared, "Training RMSE" = RMSE, "N predictors" = k)
stargazer(eval, summary = F, out=paste(output,"ch13-table-4-bicrmse.tex",sep=""), digits=2, float = F, no.space = T)

#################################################################
# Cross-validation

# set number of folds
k <- 4

set.seed(13505)
cv1 <- train(model1, cps, method = "lm", trControl = trainControl(method = "cv", number = k))
set.seed(13505)
cv2 <- train(model2, cps, method = "lm", trControl = trainControl(method = "cv", number = k))
set.seed(13505)
cv3 <- train(model3, cps, method = "lm", trControl = trainControl(method = "cv", number = k), na.action = "na.omit")
set.seed(13505)
cv4 <- train(model4, cps, method = "lm", trControl = trainControl(method = "cv", number = k), na.action = "na.omit")
set.seed(13505)


# calculate average rmse
cv <- c("cv1", "cv2", "cv3", "cv4")
rmse_cv <- c()

for(i in 1:length(cv)){
  rmse_cv[i] <- sqrt((get(cv[i])$resample[[1]][1]^2 +
                        get(cv[i])$resample[[1]][2]^2 +
                        get(cv[i])$resample[[1]][3]^2 +
                        get(cv[i])$resample[[1]][4]^2)/4)
}


# summarize results
cv_mat <- data.frame(
  rbind(cv1$resample[4, ], "Average"),
  rbind(cv1$resample[1, ], rmse_cv[1]),
  rbind(cv2$resample[1, ], rmse_cv[2]),
  rbind(cv3$resample[1, ], rmse_cv[3]),
  rbind(cv4$resample[1, ], rmse_cv[4])
)

colnames(cv_mat)<-c("Resample","Model1", "Model2", "Model3", "Model4")
cv_mat
