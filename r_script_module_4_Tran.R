# Trang Tran, ALY6015, Module 4 Practice, May 14

cat("\014")  # clears console
rm(list = ls())  # clears global environment
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE) # clears plots
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE) # clears packages
options(scipen = 100) # disables scientific notion for entire R session

library(pacman)
p_load(tidyverse, caret, ISLR, skimr, glmnet)

# import the data
attach(College)
head(College)
colnames(College)

# Custom function to calculate RMSE
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Split the data into a train and test set 
set.seed(123)  # Set a seed for reproducibility
train_index <- sample(1:nrow(College), nrow(College) * 0.7)  # 70% for training
train_df <- College[train_index, ]
test_df <- College[-train_index, ]

writeLines('\n*********************')
writeLines('split out design matrix and target vector from train_df:')
train_x <- model.matrix(Grad.Rate ~ ., train_df)[, -1]
train_y <- train_df$Grad.Rate

writeLines('\n*********************')
writeLines('split out design matrix and target vector from test_df:')
test_x <- model.matrix(Grad.Rate ~ ., test_df)[, -1]
test_y <- test_df$Grad.Rate

################################################################################
############## Ridge Regression ################################################

# Estimate lambda.min and lambda.1se using cross-validation
alpha <- 0 # alpha = 0 for Ridge
cv <- cv.glmnet(train_x, train_y, nfolds = 10, alpha = alpha)

# Plot the results 
plot(cv)

lambda_min <- cv$lambda.min
lambda_1se <- cv$lambda.1se
log(lambda_min)
log(lambda_1se)

# Fit a Ridge regression model against the training set

############# model 1: lambda.1se and alpha = 0
model_1se <- glmnet(train_x, train_y, alpha = alpha, lambda = cv$lambda.1se)

# print coefficients of model
print(coef(model_1se))

# eval model on train set
preds_1se <- predict(model_1se, newx = train_x)
rmse_train <- rmse(train_df$Grad.Rate, preds_1se)

# eval model on test set
preds_1se <- predict(model_1se, newx = test_x)
rmse_test <- rmse(test_df$Grad.Rate, preds_1se)

############# model 2: lambda.min and alpha = 0
model_min <- glmnet(train_x, train_y, alpha = alpha, lambda = cv$lambda.min)

# print coefficients of model
print(coef(model_min))

# eval model on train set
preds_min <- predict(model_min, newx = train_x)
rmse_train <- rmse(train_df$Grad.Rate, preds_min)

# eval model on test set
preds_min <- predict(model_min, newx = test_x)
rmse_test <- rmse(test_df$Grad.Rate, preds_min)

################################################################################
############## LASSO Regression ################################################

# Estimate lambda.min and lambda.1se using cross-validation
alpha <- 1 # alpha = 1 for LASSO
cv <- cv.glmnet(train_x, train_y, nfolds = 10, alpha = alpha)

# Plot the results 
plot(cv)

lambda_min <- cv$lambda.min
lambda_1se <- cv$lambda.1se
log(lambda_min)
log(lambda_1se)

# Fit a LASSO regression model against the training set

############# model 3: lambda.1se and alpha = 1
model_1se <- glmnet(train_x, train_y, alpha = alpha, lambda = cv$lambda.1se)

# print coefficients of model
print(coef(model_1se))

# eval model on train set
preds_1se <- predict(model_1se, newx = train_x)
rmse_train <- rmse(train_df$Grad.Rate, preds_1se)

ggplot(train_df, aes(x= preds_1se, y= train_df$Grad.Rate)) +
  geom_point() +
  geom_abline(intercept=0, slope=1) +
  labs(x='Predicted', y='Actuals', title='Train: Predicted vs. Actuals using lambda.1se LASSO')

# eval model on test set
preds_1se <- predict(model_1se, newx = test_x)
rmse_test <- rmse(test_df$Grad.Rate, preds_1se)

ggplot(test_df, aes(x= preds_1se, y= test_df$Grad.Rate)) +
  geom_point() +
  geom_abline(intercept=0, slope=1) +
  labs(x='Predicted', y='Actuals', title='Test: Predicted vs. Actuals using lambda.1se LASSO')

############# model 4: lambda.min and alpha = 1
model_min <- glmnet(train_x, train_y, alpha = alpha, lambda = cv$lambda.min)

# print coefficients of model
print(coef(model_min))

# eval model on train set
preds_min <- predict(model_min, newx = train_x)
rmse_train <- rmse(train_df$Grad.Rate, preds_min)

ggplot(train_df, aes(x= preds_min, y= train_df$Grad.Rate)) +
  geom_point() +
  geom_abline(intercept=0, slope=1) +
  labs(x='Predicted', y='Actuals', title='Train: Predicted vs. Actuals using lambda. LASSO')

# eval model on test set
preds_min <- predict(model_min, newx = test_x)
rmse_test <- rmse(test_df$Grad.Rate, preds_min)

ggplot(test_df, aes(x= preds_min, y= test_df$Grad.Rate)) +
  geom_point() +
  geom_abline(intercept=0, slope=1) +
  labs(x='Predicted', y='Actuals', title='Test: Predicted vs. Actuals using lambda.min LASSO')

################################################################################
############## Comparison ################################################
# Stepwise selection method
step_output <- step(lm(Grad.Rate ~ ., data = train_df), direction = 'both')

print(step_output$call)

# step_output lm() call - fit the model from step
model_both <- lm(formula = Grad.Rate ~ Private + Apps + Accept + Top25perc + 
                   P.Undergrad + Outstate + Room.Board + Personal + PhD + Terminal + 
                   perc.alumni + Expend, data = train_df)

# check the model out
print(summary(model_both))

# make predictions with the model on the test set
preds_model_both <- predict(model_both, new = test_df)

# get the rmse of the predictions
rmse_test <- rmse(test_df$Grad.Rate, preds_model_both)

# compare best model from glmnet work with the model from feature selection
best_glmnet_model_preds <- preds_1se

# get absolute error for model_both
ae_model_both <- abs(preds_model_both - test_df$Grad.Rate)

# get absolute error for best_glmnet_model
ae_best_glmnet_model <- abs(best_glmnet_model_preds - test_df$Grad.Rate)

# create a data frame for testing
model_df <- data.frame(
  'ae' = c(unname(ae_model_both), unname(ae_best_glmnet_model)), 
  'model' = c(rep('model_both', length(preds_model_both)), 
              rep('best_glmnet_model', length(best_glmnet_model_preds))))

# check out normality of ae for model_both
hist(model_df[model_df$model == 'model_both', 'ae'])
print(shapiro.test(model_df[model_df$model == 'model_both', 'ae']))

# check out normality of ae for best_glmnet_model
hist(model_df[model_df$model == 'best_glmnet_model', 'ae'])
print(shapiro.test(model_df[model_df$model == 'best_glmnet_model', 'ae']))

# the data is not normally distributed - check out the mean and standard 
# deviation of the ae of the two models

writeLines('\n***************************')
writeLines('Check out the means of the ae of the two models')
print(aggregate(model_df$ae, by = list(model_df$model), FUN = mean))

writeLines('\n***************************')
writeLines(paste('Check out the standard deviations (sqrt(variance)) of the ae',
                 'of the two models'))
print(aggregate(model_df$ae, by = list(model_df$model), FUN = sd))

writeLines('\n***************************')
# sneak preview - use non parametric method
# state hypotheses, claim and level of significance - use p-value method
# H0: means are equal
# H1: means are not equal
# claim: H1
# level of significance = 0.05
x <- model_df[model_df$model == 'best_glmnet_model', 'ae']
y <- model_df[model_df$model == 'model_both', 'ae']
print(wilcox.test(x, y, alternative = "two.sided"))
# we fail to reject H0 and summarize that there is not enough evidence to 
# support the claim

# plot mean ae as box plots
boxplot(ae ~ model, data = model_df)


