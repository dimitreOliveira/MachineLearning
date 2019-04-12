### Estimation Maximization

Hello Wizards!!! Today we are going to learn about Expectation Maximization. This is advanced statistics and I would recommend you guys to refresh your math on probability distribution and read about maximum likelihood(MLE) and maximum posteriori (MAP). I bet you have come across k-means algorithm and it is a famous variation of expectation maximization algorithm. Before we start with Expectation Maximization let us start with latent variable models and then Gaussian Mixture Models.

![](https://www.theschool.ai/wp-content/uploads/2019/03/2wapcm.jpg)

### Latent Variable Model  – https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-latent-erice-99.pdf

LVM is a stat method consists of two different variables:

Observed variables: Can be measured

Latent variables: Hidden variables which cannot be directly measured but can be inferred from other observed variables.

The main purpose of latent variables is to model more concepts in the data that can be easily understood but not observed. Adding latent variables will simplify the model by reducing the number of parameters we have to estimate.

![](https://www.theschool.ai/wp-content/uploads/2019/03/2wax54.jpg)

Let’s take the global problem of global warming and try to model the outcomes such as increase in humidity, increase in sea level and climate changes (observed outcomes) and caused by air pollution, deforestation, increase in population( observed inputs). These problems are interlinked and have a very high correlation between each other. So we can model this problem as having mediating factors causing a non-observable hidden variable ‘Global Warming’, which in turn causes outcomes.

![](https://www.theschool.ai/wp-content/uploads/2019/03/Picture2-768x421.png)

Notice that the number of connections now grows linearly not exponentially as we add latent factors, this greatly reduces the number of parameters you have to estimate. In general, you can have an arbitrary number of connections between variables with as many latent variables as you wish. These models are more generally known as [Probabilistic graphical models (PGMs)](https://en.wikipedia.org/wiki/Graphical_model). One of the simplest kinds of PGMs is when you have a 1-1 mapping between your latent variables (usually represented by zi) and observed variables (xi), and your latent variables take on discrete values (zi∈1,…,K).

### Gaussian Mixture Models

As an example, suppose we’re trying to understand the prices of houses across the city. The housing price will be heavily dependent on the neighborhood, that is, houses clustered around a neighborhood will be close to the average price of the neighborhood. In this context, it is straight forward to observe the prices at which houses are sold (observed variables) but what is not so clear is how is to observe or estimate the price of a “neighborhood” (the latent variables). A simple model for modeling the neighborhood price is using a Gaussian (or normal) distribution, but which house prices should be used to estimate the average neighborhood price? Should all house prices be used in an equal proportion, even those on the edge? What if a house is on the border between two neighborhoods? Can we even define clearly if a house is in one neighborhood or the other? These are all great questions that lead us to a particular type of latent variable model called a Gaussian mixture model. Visually, we can imagine the density of the observed variables (housing prices) as the “sum” or mixture of several Gaussians.

![](https://www.theschool.ai/wp-content/uploads/2019/03/gmm-768x512.png)

GMM is a probability distribution that consists of multiple probability distributions.

![](https://www.theschool.ai/wp-content/uploads/2019/03/687474703a2f2f692e696d6775722e636f6d2f46384466316d332e706e67-768x522.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/687474703a2f2f692e696d6775722e636f6d2f3735304e7363572e706e67-768x515.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/687474703a2f2f342e62702e626c6f6773706f742e636f6d2f2d7a75435142724e383939302f5647743435505a485868492f41414141414141414131452f6a74515161416a2d504d632f73313630302f676d6d322e706e67.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/687474703a2f2f692e696d6775722e636f6d2f30765a67364e582e706e67-768x474.png)

### What is Expectation–Maximization (EM) algorithm?

An iterative method for finding maximum likelihood (MLE) or maximum a posteriori (MAP) estimates of parameters in statistical models, when the model depends on unobserved latent variables

![](https://www.theschool.ai/wp-content/uploads/2019/03/Picture1-768x957.png)

### Expectation–Maximization (EM) Workflow

Alternates between performing:

- **Expectation (E)**step: Given the current parameters of the model, estimate a probability distribution.
- **Maximization (M)**step: Given the current data, estimate the parameters to update the model.

### EM, more formally

Alternates between performing:

- **Expectation (E)**step: Using the current estimate for the parameters, create function for the expectation of the log-likelihood.
- **Maximization (M)**step: Computes parameters maximizing the expected log-likelihood found on the E step.
The M parameter-estimates are then used to determine the distribution of the latent variables in the next E step.

![](https://www.theschool.ai/wp-content/uploads/2019/03/687474703a2f2f692e696d6775722e636f6d2f423548677872482e706e67-768x429.png)

### EM is trying to maximize the following function:

- X is directly observed variable
- θ parameters of model
- Z is not directly observed / latent variable
- Z is a joint (related) distribution on x.

### EM Steps

1. Initialize the parameters θ
2. Compute the best values for Z given θ
3. Use the computed values of Z to compute a better estimate for the θ
4. Iterate steps 2 and 3 until convergence

### EM steps stated another way

1. Initialize the parameters of the models, either randomly or doing a “smart seeding”
2. E Step: Find the posterior probabilities of the latent variable given current parameter values.
3. M Step: Re-estimate the parameter values given the current posterior probabilities.
4. Repeat 2-3 monitoring the likelihood function likelihood. Hope for convergence.
 
![](https://www.theschool.ai/wp-content/uploads/2019/03/687474703a2f2f692e696d6775722e636f6d2f6b6244323343762e6a7067.jpeg)

A sample implementation of EM

![](https://www.theschool.ai/wp-content/uploads/2019/03/Screen-Shot-2019-03-18-at-12.48.51-AM.png)
 

Like always we can learn only when we try coding it yourself. Choose your favorite problem and apply EM on it and share the results with us.

Happy learning Wizards.. See you next week.

Credits:

http://bjlkeng.github.io/posts/the-expectation-maximization-algorithm/

https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html

https://github.com/mcdickenson/em-gaussian

https://github.com/llSourcell/Gaussian_Mixture_Models