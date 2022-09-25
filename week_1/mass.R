#source https://datascienceplus.com/linear-regression-from-scratch-in-r/

library(MASS)
str(Boston)

y <- Boston$medv

# Matrix of feature variables from Boston
X <- as.matrix(Boston[-ncol(Boston)])

# vector of ones with same length as rows in Boston
int <- rep(1, length(y))

# Add intercept column to X
X <- cbind(int, X)

# Implement closed-form solution
betas <- solve(t(X) %*% X) %*% t(X) %*% y

# Round for easier viewing
betas <- round(betas, 2)

print(betas)

#"Notice that one of our features, ‘chas’, is a dummy variable 
#which takes a value of 0 or 1 depending on whether or not the tract
# is adjacent to the Charles River. The coefficient of ‘chas’ tells us that homes in tracts adjacent to the Charles River (coded as 1) have a median price that is $2,690 higher than homes in tracts that do not border 
#the river (coded as 0) when the other variables are held constant." https://datascienceplus.com/linear-regression-from-scratch-in-r/

# Linear regression model
lm.mod <- lm(medv ~ ., data=Boston)

# Round for easier viewing
lm.betas <- round(lm.mod$coefficients, 2)

# Create data.frame of results
results <- data.frame(our.results=betas, lm.results=lm.betas)

print(results)
