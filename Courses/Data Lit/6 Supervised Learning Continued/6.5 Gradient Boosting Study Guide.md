### Gradient Boosting Study Guide
#### Data-Lit Week 6
##### Carson Bentley       March 4, 2019
#####Gradient Boosting Study Guide

#### Decision Trees

At the heart of XGBoost is the decision tree. A decision tree is a very rudimentary model represented visually by a branching tree. Each node holds a question with two options.  The branches terminate at ‘leaves’ which represent final decisions. Training consists of adjusting the threshold at each decision node. One problem with decision trees is that they tend to overfit the training data. Decision trees stand out as the easiest model to interpret, since you can easily eyeball the decision thresholds to see how the model produced a particular result.

#### AdaBoost

AdaBoost, short for adaptive boosting algorithm, was developed in 1995 in an effort to produce a method that would take many weak hypothesis and produce a strong one. This algorithm uses many single layer decision trees, called decision stumps. Each decision stump is given a weight, and the collective weighted decisions are combined to produce the final result. Each individual tree is referred to as a ‘learner’. Training takes part in a stage-wise process, in which the past learners’ progress is frozen each time a new learner is added. The resulting method performs well at binary classification.

#### Gradient Boosting Machine

The ‘gradient boosting machine’ emerged as a generalized form of AdaBoost in the late 90s. This allowed for a broader set of loss functions including regression, multi-class classification, and others. One thing you should note about GBMs is that they have a number of hyperparameters which are important to tune in order to get good results.

#### XGBoost

XGBoost stands out as one of the top algorithms used to win Kaggle competitions.

As a member of the GBM family, it can be used for both classification and regression tasks. Although it has many applications, there are certain circumstance when it is not the best choice. In particular XGBoost is not well suited for natural language processing and computer vision tasks, where neural networks are commonly used instead. As a rule of thumb, it’s safe to use XGBoost when you have more than 1000 training examples and less than 100 features.

#### Cross Validation

In regular validation, we section off a portion of the training set to evaluate how well our model is doing over a series of training iterations. Unfortunately, our model can become overly optimistic with regards to the validation data, resulting in a gap in performance when we bring it into the world of testing. In ‘k-fold cross validation’, we split the training data into a series of ‘folds’, and at each stage of the process we set aside a different fold to use as validation. This method is useful for training both GBMs as well as neural networks.