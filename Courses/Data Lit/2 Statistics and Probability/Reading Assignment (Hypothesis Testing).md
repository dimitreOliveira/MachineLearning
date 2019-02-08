## Reading Assignment (Hypothesis Testing)

## Data-Lit Week 2

Carson Bentley       Feb 4, 2019
 

[Think Stats Chapter 7 – Hypothesis Testing](http://greenteapress.com/thinkstats/html/thinkstats008.html) 20 minutes

This article gives a solid introduction to the topic.
 

[Scipy Lecture Notes 3.1. Statistics in Python – Hypothesis Testing: Comparing Two Groups](http://scipy-lectures.org/packages/statistics/index.html#hypothesis-testing-comparing-two-groups) 7 minutes

Read this short section on one and two sample t-tests for insight on how to make use of scipy’s pre-built functions for statistics.

I’ve additionally put together [a colab notebook](https://colab.research.google.com/github/aztecman/DataLit/blob/master/Hypothesis_Testing_Demo_DataLit_Week_2.ipynb) as a complement to the article.


[Python for Data Analysis Part 24: Hypothesis Testing and the T-Test](http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-24.html) 15 minutes

Similar to the article above, this jupyter notebook further helps to explain some of the useful statistics-related functions built into the scipy library.


[Statistical Distributions](http://people.stern.nyu.edu/adamodar/New_Home_Page/StatFile/statdistns.htm) 20-30 minutes

This gives an introduction to the broader variety of distributions that can occur.


[AP Statistics: Significance Tests (Hypothesis Testing)](https://www.khanacademy.org/math/ap-statistics/tests-significance-ap/modal/v/idea-behind-hypothesis-testing) 1 hour (or more) of quality video content

For those interested in understanding the math, I’d recommend checking out the full AP Statistics coarse offered on KhanAcademy.


### Main Concepts:

#### Standard Deviation

Standard deviation gives a measure of how spread out from the center a group of data points is. Together with the mean (average), this number can be used to fully describe a normal probability distribution. The word ‘normal’ is used here in a math sense, for a distribution of data aligned to a bell curve. Increasing the sample size results in a more normal distribution.

#### Null Hypothesis:

When a scientist wants to test for an effect (ie that one drug is more effective than another), they frame the problem in reverse. That is, rather than test for a difference, the scientist tests for sameness. If the scientist can disprove the sameness of the two drugs (the null hypothesis), they can confidently say that one is more effective than the other.

#### Alternative Hypothesis:

The alternative hypothesis is simply the opposite of the null hypothesis. This is used to represent the condition when two things are different, such as when one clinical treatment is more effective. Generally, this is what the scientist wants to prove.

#### Significance (alpha):

In order to decide if a result is meaningful, it is necessary to choose a threshold called ‘significance’. This number defines how extreme a result must be to cause us to doubt our null hypothesis. Usually the value of 0.05 (5%) is chosen for our significance. This can also be described as a confidence interval of 95%.

#### P-Value:

The p-value is the probability of observing an effect (test statistic) as extreme as what was observed, under the assumption that the null hypothesis is true. This is the value which is compared to the significance, in order to judge the validity of the results.

#### Test Statistic:

A test statistic is a single random number, obtained by sampling observed data as an intermediate step to getting the p-value. It goes by various names (z-statistic / z-value, t-statistic, f-statistic, chi-square statistic) depending on the context. In the case of a t or z-test, a value close to zero means that the sample is evidence toward failing to reject the null hypothesis.

#### Type I error:

A type-1 error occurs when we reject the null hypothesis, but in reality it is true. This represents the case when we test for a difference and it exceeds our significance (alpha), but is in fact merely due to random chance. This error occurs with a chance equal to our P-value.

#### Type II error:

In the case of a type-2 error, we fail to reject the null hypothesis when actually we should reject it. We avoid making a type-2 error with a probability equal to our ‘power’.

#### Power:

‘Power’ is the probability of rejecting the null hypothesis when it is false. Since power is the same as the chance of a not making a type-2 error, more power means higher quality scientific evidence. We can get more power in our results by either increasing alpha (at the cost of type-1 errors), or by increasing the amount of data used in our sampling process.

#### Z-Test

The z-test is used when we are comparing proportions. The final value returned, called the z-value, is a test statistic that measures how many standard deviations the observed sample proportion is from the norm.

#### T-Test

A t-test is very similar to a z-test, but it is used when comparing means, rather than proportions.

#### One Sample vs Two Sample Test

When we are comparing to an assumption, or a predefined standard, we use a one sample test. On the other hand, when we have two populations from which to draw samples, as is the case for an A/B test, we do a two sample test.