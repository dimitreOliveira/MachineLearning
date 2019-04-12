## Discrete Probability


In this blog, I mainly cited
* [Stat Trek](https://stattrek.com/probability-distributions/discrete-continuous.aspx)
* [study.com](https://study.com/academy/lesson/discrete-probability-distributions-equations-examples.html)
* [wikipedia](https://en.wikipedia.org/wiki/Bernoulli_distribution)


# Discrete vs. Continuous
All probability distributions can be classified as continuous probability distributions or as discrete probability distributions.
If a variable can take on any value between two specified values, it is called a continuous variable; otherwise, it is called a discrete variable.

# Discrete Probability Distributions
If a random variable is discrete, its probability distribution is called a discrete probability distribution.
Suppose you are a fan of Pokemon Go. Of course, it doesn’t matter if you do’t like Pokemon Go.
![](https://www.theschool.ai/wp-content/uploads/2019/02/pokemon-768x281.png)

As a fan, you want to catch a lot of Pikachu, so you counted caught Pokemon every day for 14 days.
Don’t be sad about less Pikachu than expected. We can predict how many Pikachu can you catch by discrete probability.
![](https://www.theschool.ai/wp-content/uploads/2019/02/table-600x696.png)

The number of Pikachu is an example of a discrete random variable because there are only certain values that are possible (30, 40, 50, etc.),
so this represents a discrete probability distribution.
![](https://www.theschool.ai/wp-content/uploads/2019/02/t2.png)

If you add up all the probabilities, you should get exactly one. This is true for all discrete probability distributions.
0.2143 + 0.1429 + 0.2857 + 0.1429 + 0.1429 + 0.0714 = 1.000

# Expected Value Function
E(x) = ∑pixi

Using the expected value function, you can calculate how many Pikachu can be caught on an average day.Simply multiply each amount 
xi  by the probability pi that you will expect to catch Pikachu in a given day.Then, add all of these value together.

E(x) = 0.2143*30 + 0.1429*40 + 0.2857*50 + 0.1429*60 + 0.1429*70 + 0.0714*80

E(x) **= 50.719**

This means that, on average, you can expect to catch about **51 Pikachu each day**! It also represents the mean of the data set.

# Variance and Standard Deviation
Variance is one way to measure the spread in a data set, and it is defined as the sum of the squared deviations from the mean. For a discrete probability distribution, variance can be calculated using the following equation:

Var(x)=∑pix2i−[E(x)]2

Var(x) = 0.2143*30^2 +0.1429*40^2 +0.2857*50^2 + 0.1429*60^2 + 0.1429*70^2 + 0.0714*80^2 – 50.719^2 **= 234.953039**

**Standard Deviation** is simply equal to the square root of the variance:

σ = √Var(x)

σ = √234.953039 = 15.32817794

There are good days and bad days. Since the standard deviation of Pikachu reaches 15, you can only catch 14 on a very unlucky day. But, Don’t sad anymore because we studied discrete probability. You can catch more than 95 Pikachu on a good day. Remember that a good day must come when you  keep going something like Pokemon or statistics steadily.

# Well-known discrete probability distributions

```
Bernoulli distribution
Binomial Distribution
Multinomial Distribution
Hypergeometric Distribution
Negative binomial Distribution
Poisson Distribution
Geometric Distribution
discrete uniform distribution
Moment-Generation
```

There are many well known types . Let’s look into Bernoulli and Poisson.

# Bernoulli Distribution

Every time you flip a coin, there are only two possible outcomes, heads or tails, and there’s a 50% chance of either outcome. A coin flip is an example of a **Bernoulli trial**, which is any random experiment in which there are exactly two possible outcomes.

A Bernoulli distribution is the probability distribution for a series of Bernoulli trials. It is the simplest kind of discrete probability distribution that can only take **two possible values, usually 0 and 1 (success and failure).** It named after Swiss mathematician Jacob Bernoulli.

![](https://www.theschool.ai/wp-content/uploads/2019/02/Jakob_Bernoulli-268x300.jpg)

It is important that the variable being measured is **both random and independent.** This means that the probability must be the same for every trial. If the probability changes, then a Bernoulli distribution will not be accurate.

### Variance for Bernoulli distribution

The probability of success for a Bernoulli trial is defined as 
**P**. The probability of failure is **1−P**. The variance of a series of Bernoulli trials is the measure of how spread out the values in the data set are. It is given by P(1 – P).

Var=P(1–P)

For the coin flip example,

Var = (0.5)(1 – 0.5) = 0.25

This means that for the coin flip experiment the variance would be 0.25.

# Poisson Distribution
Pronunciation (French ​[pwasɔ̃];  English  /ˈpwɑːsɒn/)
It named after French mathematician Siméon Denis Poisson in 1837.
![](https://www.theschool.ai/wp-content/uploads/2019/02/Simeon_Poisson-256x300.jpg)

It is used to calculate the probability of an event occurring over a certain interval. The **interval** can be one of time, area, volume or distance. Usually, it can be applied to systems with a **large number of possible events**, each of which is **rare.**

(λke−λ)/k!

where

k = 0,1,2,3…(number of successes for the event)

e = 2.71828 (Euler’s constant)

λ = mean number of successes in the given time interval or region of space

### Two conditions of Poisson distribution

1. Each successful event must be independent.
2. The probability of success over a short interval must equal the probability of success over a longer interval.

### Applications
* The number of deaths by horse kicking in the Prussian army (first application)
* Birth defects and genetic mutations
* Rare diseases (like Leukemia, but not AIDS because it is infectious and so not independent) – especially in legal cases
* Car accidents
* Traffic flow and ideal gap distance
* Number of typing errors on a page
* Hairs found in McDonald’s hamburgers
* Spread of an endangered animal in Africa
* Failure of a machine in one month
* The number of meteorites greater than 1 meter diameter that strike Earth in a year
* The number of patients arriving in an emergency room between 10 and 11 pm
* The number of photons hitting a detector in a particular time interval

https://www.intmath.com/counting-probability/13-poisson-probability-distribution.php

### Expected value and Variance of Poisson distribution 

E(x) = Var(x)=λ

Is the expected value equal to the variance and 
λ
? Intuitively, it may not be understandable, but it can be proved by using the extremes and the Taylor series. If you are curious to know the proof, please refer to the link provided below.

[Expectation_of_Poisson_Distribution](https://proofwiki.org/wiki/Expectation_of_Poisson_Distribution)

[Variance_of_Poisson_Distribution](https://proofwiki.org/wiki/Variance_of_Poisson_Distribution)

### Example
A social media w
ebsite gains an average of 12 new followers each day. Find the probability of gaining 75 followers in a week.

**1. Condition check**
* followers are independent each other
* the probability is same

**2. Define variables**
k = 75 (We are looking for 75 followers, so 75 is our number of successes.)
λ = 12*7=84 (84 new followers in a week)

**3. Poisson formula**
(84^75 e^−84)/75! = 0.028

The likelihood of getting 75 new followers in a week is **2.8%.**

https://study.com/academy/lesson/poisson-distribution-definition-formula-examples.html

# Summary
We looked at the discrete distribution and the two representative distributions of Bernoulli and Poisson.Try to remember the conditions and formulas of each distribution and practice applying them to solve the problem.

### Discrete distribution
E(x) = ∑pixi

Var(x) = ∑pix2i−[E(x)]2

σ = √Var(x)

### Bernoulli Distribution
Var = P(1–P)

### Poisson Distribution
(λke−λ)k!

E(x) = Var(x)=λ