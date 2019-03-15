### Introduction to Unsupervised Learning

![](https://blog.algorithmia.com/wp-content/uploads/2018/04/Machine-Learning.png)
http://prooffreaderswhimsy.blogspot.com/2014/11/machine-learning.html

What do you do when your dataset doesn’t have any labels? Unsupervised learning is a group of machine learning algorithms and approaches that work with this kind of “no-ground-truth” data. This post will walk through what unsupervised learning is, how it’s different than most machine learning, some challenges with implementation, and resources for further reading.

### What is Unsupervised Learning?
The easiest way to understand what’s going on here is to think of a test. When you took tests in school, there were questions and answers; your grade was determined by how close your answers were to the actual ones (or the answer key). But imagine if there was no answer key, and there were only questions. How would you grade yourself?

Now apply this framework to machine learning. Traditional datasets in ML have labels (think: the answer key), and follow the logic of “X leads to Y.” For example: we might want to figure out if people with more Twitter followers typically make higher salaries. We think that our input (Twitter followers) might lead to our output (salary), and we try to approximate what that relationship is.

![](https://blog.algorithmia.com/wp-content/uploads/2018/04/unsupervised.png)

The stars are data points, and machine learning works on creating a line that explains how the input and outcomes are related. But in unsupervised learning, there are no outcomes! We’re just looking to analyze in the input, which is our Twitter followers. There is no salary, or Y, involved at all. Just like there not being an answer key for the test.

![](https://blog.algorithmia.com/wp-content/uploads/2018/04/unsupervised2.png)

Maybe we don’t have access to salary data, or we’re just interested in different questions. It doesn’t matter! The important thing is that there is no output to match to, and no line to draw that represents a relationship.

So what exactly is the goal of unsupervised learning then? What do we do when we only have input data without labels?

### Types of Unsupervised Learning

#### Clustering

Any business needs to focus on understanding customers: who they are and what’s driving their purchase decisions?

You’ll usually have different groups of users that can be split across a few criteria. These criteria can be as simple, such as age and gender, or as complex as persona and purchase process. Unsupervised learning can help you accomplish this task automatically.

Clustering algorithms will run through your data and find these natural clusters if they exist. For your customers, that might mean one cluster of 30-something artists and another of millennials who own dogs. You can typically modify how many clusters your algorithms looks for, which lets you adjust the granularity of these groups. There are a few different types of clustering you can utilize:

- [K-Means Clustering](https://algorithmia.com/algorithms/pappacena/kmeans) – clustering your data points into a number (K) of mutually exclusive clusters. A lot of the complexity surrounds how to pick the right number for K.
- [Hierarchical Clustering](https://algorithmia.com/algorithms/weka/WekaHierarchicalClusterer) – clustering your data points into parent and child clusters. You might split your customers between younger and older ages, and then split each of those groups into their own individual clusters as well.
- [Probabilistic Clustering](https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/) – clustering your data points into clusters on a probabilistic scale.

These variations on the same fundamental concept might look something like this in code:

```
#Import the KMeans package from Scikit Learn

from sklearn.cluster import KMeans

#Grab the training data

x = os.path(‘train’)

#Set the desired number of clusters

k = 5

#Run the KMeans algorithm

kmeans = KMeans(n_clusters=k).fit(x)

#Show the resulting labels

kmeans.labels_
```

Any clustering algorithm will typically output all of your data points and the respective clusters to which they belong. It’s up to you to decide what they mean and exactly what the algorithm has found. As with much of data science, algorithms can only do so much: value is created when humans interface with outputs and create meaning.

#### Data Compression
Even with major advances over the past decade in computing power and storage costs, it still makes sense to keep your data sets as small and efficient as possible. That means only running algorithms on necessary data and not training on too much. Unsupervised learning can help with that through a process called dimensionality reduction.

Dimensionality reduction (dimensions = how many columns are in your dataset) relies on many of the same concepts as [Information Theory](http://cosmicfingerprints.com/information-theory-made-simple/): it assumes that a lot of data is redundant, and that you can represent most of the information in a data set with only a fraction of the actual content. In practice, this means combining parts of your data in unique ways to convey meaning. There are a couple of popular algorithms commonly used to reduce dimensionality:

- Principal Component Analysis (PCA) – finds the linear combinations that communicate most of the variance in your data.
- Singular-Value Decomposition (SVD) – factorizes your data into the product of three other, smaller matrices.

These methods as well as some of their more complex cousins all rely on concepts from linear algebra to break down a matrix into more digestible and informatory pieces.

Reducing the dimensionality of your data can be an important part of a good machine learning pipeline. Take the example of an image-centerpiece for the burgeoning field of computer vision. [We outlined here](https://blog.algorithmia.com/introduction-to-computer-vision/) how big a dataset of images can be and why. If you could reduce the size of your training set by an order of magnitude, that will significantly lower your compute and storage costs while making your models run that much faster. That’s why PCA or SVD are often run on images during preprocessing in mature machine learning pipelines.

### Unsupervised Deep Learning

Unsurprisingly, unsupervised learning has also been extended to neural nets and deep learning. This area is still nascent, but one popular application of deep learning in an unsupervised fashion is called an [Autoencoder](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/).

Autoencoders follow the same philosophy as the data compression algorithms above––using a smaller subset of features to represent our original data. Like a Neural Net, an Autoencoder uses weights to try and mold the input values into a desired output; but the clever twist here is that the output is the same thing as the input! In other words, the Autoencoder tries to figure out how to best represent our input data as itself, using a smaller amount of data than the original.

![](https://blog.algorithmia.com/wp-content/uploads/2018/04/word-image-2.png)
http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/

Autoencoders have proven useful in computer vision applications like object recognition, and are being researched and extended to domains like audio and speech.

### Challenges in Implementing Unsupervised Learning

In addition to the regular issues of finding the right algorithms and hardware, unsupervised learning presents a unique challenge: it’s difficult to figure out if you’re getting the job done or not.

In supervised learning, we define metrics that drive decision making around model tuning. Measures like precision and recall give a sense of how accurate your model is, and parameters of that model are tweaked to increase those accuracy scores. Low accuracy scores mean you need to improve, and so on.

Since there are no labels in unsupervised learning, it’s near impossible to get a reasonably objective measure of how accurate your algorithm is. In clustering for example, how can you know if K-Means found the right clusters? Are you using the right number of clusters in the first place? In supervised learning we can look to an accuracy score; here you need to get a bit more creative.

A big part of the “will unsupervised learning work for me?” question is totally dependent on your business context. In our example of customer segmentation, clustering will only work well if your customers actually do fit into natural groups. One of the best (but most risky) ways to test your unsupervised learning model is by implementing it in the real world and seeing what happens! Designing an A/B test–with and without the clusters your algorithm outputted–can be an effective way to see if it’s useful information or totally incorrect.

Researchers have also been working on algorithms that might give a more objective measure of performance in unsupervised learning. Check out the below section for some examples.

#### Reading and Papers

[Machine Learning For Humans – Unsupervised Learning](https://medium.com/machine-learning-for-humans/unsupervised-learning-f45587588294) – “How do you find the underlying structure of a dataset? How do you summarize it and group it most usefully? How do you effectively represent data in a compressed format? These are the goals of unsupervised learning, which is called “unsupervised” because you start with unlabeled data(there’s no Y).”

[Unsupervised Learning and Data Clustering](https://towardsdatascience.com/unsupervised-learning-and-data-clustering-eeecb78b422a) – “In some pattern recognition problems, the training data consists of a set of input vectors x without any corresponding target values. The goal in such unsupervised learning problems may be to discover groups of similar examples within the data, where it is called clustering, or to determine how the data is distributed in the space, known as density estimation.”

[Towards Principled Unsupervised Learning](https://arxiv.org/abs/1511.06440) – “General unsupervised learning is a long-standing conceptual problem in machine learning. Supervised learning is successful because it can be solved by the minimization of the training error cost function. In this paper, we present an unsupervised cost function which we name the Output Distribution Matching (ODM) cost, which measures a divergence between the distribution of predictions and distributions of labels.”

[Unsupervised Learning of Visual Representations using Videos](https://arxiv.org/abs/1505.00687) – “Is strong supervision necessary for learning a good visual representation? Do we really need millions of semantically-labeled images to train a Convolutional Neural Network (CNN)? In this paper, we present a simple yet surprisingly powerful approach for unsupervised learning of CNN. Specifically, we use hundreds of thousands of unlabeled videos from the web to learn visual representations.”

[Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681) – “We use multilayer Long Short Term Memory (LSTM) networks to learn representations of video sequences. Our model uses an encoder LSTM to map an input sequence into a fixed length representation. This representation is decoded using single or multiple decoder LSTMs to perform different tasks, such as reconstructing the input sequence, or predicting the future sequence.”

#### Tutorials

[Stanford Deep Learning Tutorial](http://deeplearning.stanford.edu/tutorial/) – “This tutorial will teach you the main ideas of Unsupervised Feature Learning and Deep Learning. By working through it, you will also get to implement several feature learning/deep learning algorithms, get to see them work for yourself, and learn how to apply/adapt these ideas to new problems.”

[Introduction to K-Means Clustering](https://www.datascience.com/blog/k-means-clustering) – “K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided.”

[Hierarchical Clustering Tutorial](http://www.econ.upf.edu/~michael/stanford/maeb7.pdf) – “In this chapter we shall consider a graphical representation of a matrix of distances which is perhaps the easiest to understand – a dendrogram, or tree – where the objects are joined together in a hierarchical fashion from the closest, that is most similar, to the furthest apart, that is the most different.”

[Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html) – “Autoencoding is a data compression algorithm where the compression and decompression functions are 1) data-specific, 2) lossy, and 3) learned automatically from examples rather than engineered by a human. Additionally, in almost all contexts where the term “autoencoder” is used, the compression and decompression functions are implemented with neural networks.”

[Principal Component Analysis in Python](https://plot.ly/ipython-notebooks/principal-component-analysis/) – “Principal Component Analysis (PCA) is a simple yet popular and useful linear transformation technique that is used in numerous applications, such as stock market predictions, the analysis of gene expression data, and many more. In this tutorial, we will see that PCA is not just a “black box”, and we are going to unravel its internals in 3 basic steps.”

#### Lectures and Videos

[Machine Learning: Unsupervised Learning (Udacity + Georgia Tech)](https://www.udacity.com/course/machine-learning-unsupervised-learning--ud741) – “Closely related to pattern recognition, Unsupervised Learning is about analyzing data and looking for patterns. It is an extremely powerful tool for identifying structure in data. This course focuses on how you can use Unsupervised Learning approaches — including randomized optimization, clustering, and feature selection and transformation — to find structure in unlabeled data.”

[Unsupervised Learning in R (Datacamp)](https://www.datacamp.com/courses/unsupervised-learning-in-r) – “Many times in machine learning, the goal is to find patterns in data without trying to make predictions. This is called unsupervised learning. This course provides a basic introduction to clustering and dimensionality reduction in R from a machine learning perspective, so that you can get from data to insights as quickly as possible.”

[The Next Frontier in AI: Unsupervised Learning (Yann LeCun)](https://www.youtube.com/watch?v=IbjF5VjniVE) – “AI systems today do not possess “common sense”, which humans and animals acquire by observing the world, acting in it, and understanding the physical constraints of it. Some of us see unsupervised learning as the key towards machines with common sense. Approaches to unsupervised learning will be reviewed. This presentation assumes some familiarity with the basic concepts of deep learning.”

[Unsupervised Learning Course Page (UCL)](http://mlg.eng.cam.ac.uk/zoubin/course05/index.html) – “This course provides students with an in-depth introduction to statistical modelling and unsupervised learning techniques. It presents probabilistic approaches to modelling and their relation to coding theory and Bayesian statistics. We will cover Markov chain Monte Carlo sampling methods and variational approximations for inference.”