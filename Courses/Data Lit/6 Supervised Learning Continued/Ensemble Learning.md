### Ensemble Learning

In this blog, I mainly cited
https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning
https://quantdare.com/what-is-the-difference-between-bagging-and-boosting
https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229

![](https://www.theschool.ai/wp-content/uploads/2019/03/ensemble_soai.jpg)
 

## Introduction

Ensemble learning is the art of combining diverse set of learners (individual models) together to improvise on the stability and predictive power of the model.

In other words, ensemble learning is a technique that uses a multi-classifier to overcome the limitations of a single classifier and maximizes the performance of a machine learning model. It is a practical skill, not a simple theory, thus you should get used to it to have an ace up your sleeve if you want to get high ranked in Kaggle and Hackathon.

## Bias vs. Variance

![](https://www.theschool.ai/wp-content/uploads/2019/03/variance_bias.png)

##### The red is the real value and the blue dots are predictions

#### Bias
* Quantifies how much on an average are the predicted values **different from the actual value**.
* A high bias error means we have a **under-fitting** model which keeps on missing important trends.

#### Variance
* Quantifies how are the prediction made on same observation **different from each other**.
* A high variance model will **over-fit** on your training population and perform badly on any observation beyond training.

![](https://www.theschool.ai/wp-content/uploads/2019/03/0.png)

### Bias-Variance Trade-off

* AIf the model is too simple and has very few parameters then it may have high bias and low variance.
* If the model has large number of parameters then it’s going to have high variance and low bias.
* This trade-off in complexity is why there is a trade-off between bias and variance.
* An algorithm can’t be more complex and less complex at the same time.

![](https://www.theschool.ai/wp-content/uploads/2019/03/The-bias-variance-tradeoff-with-increasing-model-complexity.jpg)
##### The bias-variance trade-off with increasing model complexity
##### https://www.researchgate.net/figure/The-bias-variance-tradeoff-with-increasing-model-complexity_fig1_279299325

![](https://www.theschool.ai/wp-content/uploads/2019/03/both.jpg) 

#### Total Error
* The all types of error can be broken down into following three components:

![](https://www.theschool.ai/wp-content/uploads/2019/03/error-of-a-model.png)

* To build a good model, we need to find a good balance between bias and variance such that it minimizes the total error.
* This is known as the trade-off management of bias-variance errors.

![](https://www.theschool.ai/wp-content/uploads/2019/03/model_complexity.png)

* Ensemble learning is one way to execute this trade off analysis.

### Ensemble learning techniques
* They can be divided into two main groups.
* There are three popular techniques: Bagging, Boosting and Stacking
1. **Ensemble methods**: using the same learning technique.
    * **Bagging**: To decrease the variance.
    * **Boosting**:To decrease the bias.
2. **Hybrid methods**: using new learning techniques.
    * **Stacking**(or Stacked Generalization)

![](https://www.theschool.ai/wp-content/uploads/2019/03/0_vuIrBhNri-Wz_HMy.png)
##### Bagging and boosting

## 1. Bagging (Bootstrap Aggregating)

![](https://www.theschool.ai/wp-content/uploads/2019/03/bagging.png)

![](https://www.theschool.ai/wp-content/uploads/2019/03/bagging_ex.png)

* Bagging tries to implement **multiple same learners** on small sample populations.
* Then it **takes a mean of all the predictions**.
* It **decreases the variance**. (reduces over-fitting)
* It may cause **under-fitting** problem.
* Example : Random forest

![](https://www.theschool.ai/wp-content/uploads/2019/03/bagging2.jpg)

## 2. Boosting

![](https://www.theschool.ai/wp-content/uploads/2019/03/boostong_ex.png)

* Boosting is an **iterative** technique which **adjust the weight** of an observation based on the **last classification**.
* If an observation was classified **incorrectly**, it tries to **increase the weight** of this observation.
    * Consecutive trees (random sample) are **fit and at every step**, the goal is to **improve the accuracy** from the prior tree.
    * Combine all the weak learners **via majority voting**.
* It **decreases the bias**.
* It may cause **over-fitting** problem.
* Example : AdaBoost, GradientBoost, **XGBoost(Extreme Gradient Boost. most popular one)**

![](https://www.theschool.ai/wp-content/uploads/2019/03/xgboost.jpg)

## 3. Stacking (Meta ensembling)
* “Two heads are better than one”
* Combine information from multiple predictive models to generate a new model.
* The stacked model (also called 2nd-level model) can outperform each of the individual models.
* Most effective when the base models are significantly different.
* It **decreases both the variance and the bias**.
* It may demand on huge computing power.
* Example : KNN+SVM

## Summary

#### Bagging vs. Boosting

![](https://www.theschool.ai/wp-content/uploads/2019/03/bb.png)
 
#### Comparison

![](https://www.theschool.ai/wp-content/uploads/2019/03/bbs.png)

## Further reading
https://www.youtube.com/watch?v=2Mg8QD0F1dQ
https://www.youtube.com/watch?v=GM3CDQfQ4sw
https://www.analyticsindiamag.com/primer-ensemble-learning-bagging-boosting
http://quantdare.com/dream-team-combining-classifiers-2
https://hackernoon.com/boosting-algorithms-adaboost-gradient-boosting-and-xgboost-f74991cad38c
http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice