### Types of Regression

In this blog, I mainly cited

[SUNIL RAY's posting](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression)

[ListenData's posting](https://www.r-bloggers.com/15-types-of-regression-you-should-know/)

[George Seif's posting](https://towardsdatascience.com/5-types-of-regression-and-their-properties-c5e1fa12d55e)

![](https://www.theschool.ai/wp-content/uploads/2019/02/JPMorgan-machine-learning-classification.jpg)
This image is from J.P Morgan

## Introduction

Regression techniques are the most popular statistical techniques used for prediction tasks. **Linear** and **Logistic** regressions are commonly used in real world  since they are easy to use and interpret.  But there are **more than 15 types of regression**, each with their own strengths and weaknesses.

Regression analysis is mainly used for

* Causal analysis
* Forecasting the impact of change
* Forecasting trends

In this post, we’re going to look at 3 of the most common types of regression algorithms and their properties of **Linear, Logistic and Ridge**.

## Types of Regression
### 1. Linear Regression

![](https://www.theschool.ai/wp-content/uploads/2019/02/Linear_Regression1.png)

* It is the simplest form of regression and always introduced at the first order.
* The dependent variable is continuous, independent variable(s) can be continuous or discrete, and nature of regression line is linear.
* It establishes a relationship between dependent variable Y and one or more independent variables X using a best fit straight line (also known as regression line).
*  In case of more than one predictors present, it is called multiple linear regression.

Y = a + b ∗ X + e

* a = slope of the line
* b = y-intercept
* e = error term

### How to obtain best fit line (Value of a and b )?

![](https://www.theschool.ai/wp-content/uploads/2019/02/reg_error.gif)

* Least Square Method is used for fitting a regression line.
* It calculates the best-fit line by minimizing the sum of the squares of the vertical deviations.

![](https://www.theschool.ai/wp-content/uploads/2019/02/Least_Square.png)

### Important Points
* There must be linear relationship between independent and dependent variables.
* It is very sensitive to outliers. It can affect the regression line and eventually the prediction.
* Multiple regression suffers from multicollinearity, auto-correlation and heteroscedastic.
* Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes. The result may be unstable.
* In case of multiple independent variables, try to use forward selection, backward elimination and step wise approach to select most significant independent variables .

### 2. Logistic Regression

![](https://www.theschool.ai/wp-content/uploads/2019/02/linear_vs_logistic_regression_edxw03.png)

This image is from https://www.imagenesmy.com/imagenes/logistic-regressopm-27.html

* It describes the relationship between a set of independent variables and a categorical dependent variable.
* It is used to find the probability of event=Success and event=Failure when the dependent variable is binary (0/ 1, True/ False, Yes/ No) in nature.
* The value of Y ranges from 0 to 1 and it can represented by following equation.

```
odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
ln(odds) = ln(p/(1-p))
logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
```
![](https://www.theschool.ai/wp-content/uploads/2019/02/logis1.jpg)
![](https://www.theschool.ai/wp-content/uploads/2019/02/logis2-768x384.png)

#### Why don’t we use linear regression in this case?
* In linear regression, range of Y is real line but here it can take only 2 values. So Y is either 0 or 1 
* But X′B is continuous thus we can’t use usual linear regression in such a situation.
* Secondly, the error terms e are not normally distributed.
* Y follows binomial distribution and hence is not normal.

#### Important Points
* It is widely used for **classification problems**.
* It **doesn’t require linear relationship** between dependent and independent variables.
* It applies a non-linear log transformation to the predicted odds ratio thus it can handle various types of relationships.
* To avoid over fitting and under fitting, you should include all significant variables.
* It requires **large sample sizes** because maximum likelihood estimates are less powerful at low sample sizes than ordinary least square
* The independent variables should not be correlated with each other i.e. **no multicollinearity**.
* If the values of dependent variable is ordinal, then it is called as **Ordinal logistic regression**.
* If dependent variable is multi class then it is known as **Multinomial Logistic regression**.

### 3. Ridge Regression (Shrinkage Regression)

![](https://www.theschool.ai/wp-content/uploads/2019/02/ridge3-768x386.png)

This image is from [Robert R.F. DeFilippi’s blog](https://medium.com/@rrfd/what-is-ridge-regression-applications-in-python-6ed3acbb2aaf)

* A standard linear or polynomial regression will fail in the case where there is high collinearity among the feature variables.  ( independent variables are highly correlated).
* By adding a degree of bias, ridge regression reduces the standard errors.
* A regression coefficient is not significant even though, theoretically, that variable should be highly correlated with Y.
* When you add or delete an X feature variable, the regression coefficients change dramatically.
* Your X feature variables have high pairwise correlations (check the correlation matrix).

#### Let’s review linear regression
* Y = a + b ∗ X + e
* Prediction errors can be decomposed into two sub components, the **biased** and the **variance**.

error term e is the value needed to correct for a prediction error between the observed and predicted value

```
y = a+ b*x+ e 
  = a+ y 
  = a+ b1x1+ b2x2+...+ e, for multiple independent variables.
```

#### How to solve the multicollinearity problem?

* It uses [shrinkage parameter](https://en.wikipedia.org/wiki/Shrinkage_estimator) λ (lambda).
* In the equation, there are two components.
    1. First one is least square term.
    2. Other one is lambda of the summation of β2 (beta- square) where β is the coefficient.
This is added to least square term in order to shrink the parameter to have a very low variance.

#### Important Points
* The assumptions of this regression is **same** as **least squared regression** except **normality is not to be assumed**.
* It shrinks the value of coefficients but **doesn’t reach to zero**, which suggests no feature selection feature.
* This is a **regularization method** and uses [l2 regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)).

## Other types of regression
* Polynomial Regression
* Quantile Regression
* Lasso Regression
* ElasticNet Regression
* Principal Component Regression
* Partial Least Square Regression
* Support Vector Regression
* Ordinal Regression
* Poisson Regression
* Negative Binomial Regression
* Quasi-Poisson Regression
* Cox Regression
Check [this article](https://www.r-bloggers.com/15-types-of-regression-you-should-know/) for more details.

### Conclusion

![](https://www.theschool.ai/wp-content/uploads/2019/02/types-768x335.png)
![](https://www.theschool.ai/wp-content/uploads/2019/02/assessment1.png)

This image is [from Mahrita Harahap’s blog](https://mahritaharahap.wordpress.com/teaching-areas/regression-analysis/multiple-regression/)

* If dependent variable is continuous and model is suffering from collinearity or there are a lot of independent variables,
    * Try PCR, PLS, ridge, lasso and elastic net regressions.
    * You can select the final model based on Adjusted r-square, RMSE, AIC and BIC.
* If you are working on count data,
    * Try poisson, quasi-poisson and negative binomial regression.
* To avoid overfitting,
    * Use cross-validation method to evaluate models used for prediction.
    * Also try ridge, lasso and elastic net regressions techniques.
* When you have non-linear model
    * Try support vector regression.