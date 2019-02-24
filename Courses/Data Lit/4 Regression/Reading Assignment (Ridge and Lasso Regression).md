### Reading Assignment (Ridge and Lasso Regression)

## Introduction

Ridge and Lasso regression are two additional tools you can use to improve results and accuracy on your linear regression models while reducing over-fitting.

When independent variables are related/correlated to each other (collinearity), this can have a negative impact on the accuracy of the model and cause it to over-fit to training data. Ridge and Lasso regression are based on simple linear regression, but tweak the mean squared error cost function to improve results.

Over-fitting happens when the model tries too hard to adapt to noise in your training data. A technique called regularization helps constrain the magnitude of the coeffecients (weights, or slope in linear equation) by adding a penalty in the cost function. Ridge and Lasso regression are two regularization techniques.

## Reading

Beginner level reading:
[A comprehensive beginners guide for Linear, Ridge and Lasso Regression](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)

Intermediate level reading:
[Regularization in Machine Learning](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

Advanced math dive with code:
[Ridge and Lasso Regression: A Complete Guide with Python Scikit-Learn](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b)

## Homework Assignment

Fork the [Kaggle notebook](https://www.kaggle.com/colinskow/linear-regression-automobile-tutorial-data-lit) from the Linear Regression with Python [video tutorial](https://www.theschool.ai/courses/data-lit/lessons/linear-regression-python/) and apply both ridge and lasso regression at the bottom using the example code above. Are you able to improve results? Post your findings in the comments below.