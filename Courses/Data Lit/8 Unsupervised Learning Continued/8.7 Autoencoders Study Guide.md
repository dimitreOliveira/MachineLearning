### Autoencoders Study Guide

Data-Lit Week 8

Carson Bentley       March 19, 2019

Study Guide
 

**Introduction**

One of the easiest unsupervised methods to understand is the autoencoder. Formalized in the mid-eighties, this neural net architecture has had nearly 40 years to develop. The basic idea is to train a network to reproduce the input information based on a compact intermediate representation. This bottleneck can be achieved either by limiting the number of nodes, or by applying a regularization term. Whatever the method used, the overall goal is to produce a representation in the learned filter weights more interesting than the identity function. The model is broken down into an encoder and decoder section which are trained together.

![](https://www.theschool.ai/wp-content/uploads/2019/03/autoencoder1.jpg)
##### from [Modular Learning in Neural Networks](https://www.aaai.org/Papers/AAAI/1987/AAAI87-050.pdf)
 

### Common Types of Autoencoders
 

**Undercomplete Autoencoder** (vanilla)

The most common kind of autoencoder is the ‘undercomplete’ version which is commonly depicted with an hourglass shape. By limiting the number of nodes in the middle most layers, the network learns a compact representation of the data. This provides a convenient way to perform dimensionality reduction. Another use for the ‘vanilla’ (densely connected rather than convolutional) autoencoder is detecting outliers, such as for fraud.

![](https://www.theschool.ai/wp-content/uploads/2019/03/autoencoder2-600x474.png)
##### from [Deep Autoencoders For Collaborative Filtering](https://towardsdatascience.com/deep-autoencoders-for-collaborative-filtering-6cf8d25bbf1d)

**Convolutional Autoencoder**

Convolutional autoencoders are, like the name sounds, autoencoders with convolutional layers in the encoder. This makes them useful for image based tasks such as noise removal, super-resolution, and automatic colorization. ‘Deconvolution’ layers are used in the decoder (in Keras this function is called Conv2DTranspose).

**Variational Autoencoder**

If you are interested in generative deep learning models, you’ve probably already heard of VAEs. Variational autoencoders are a popular alternative to generative adversarial networks (GANs) which can be used to generate data for semi-supervised learning (in which we have a mixture of labeled and unlabeled data) as well as artistic applications. The encoder generates two vectors for mean and standard deviation, rather than the usual single vector. The decoder is used independently at the testing phase to perform generation of new data. Generally, VAEs use an undercomplete architecture. If you are interested in this approach, be sure to check out [this video](https://www.youtube.com/watch?v=9zKuYvjFFS8) by Xander of Arxiv Insights. [[paper](https://arxiv.org/abs/1312.6114)] 

**Sparse Autoencoder**

A sparse autoencoder creates a bottleneck by enforcing a penalty on hidden node activations in the loss function, rather than by directly limiting the number of nodes in intermediate layers. This means that it can be overcomplete (intermediate layers larger than the input) without risk of learning the identity function.

**Denoising Autoencoder**

Denoising happens to be another technique that allows for an overcomplete model to succeed. Given a piece of clean data (such as an image), we create a copy and add noise to it. Then, we train the network to recreate the clean data based on the noisy version.

**Contractive Autoencoders**

Contractive Autoencoders are yet another approach to overcomplete models. At a technical level, a penalty to the Frobenius norm of the Jacobian matrix of the encoder mapping is included in the loss function. At a practical level, CAEs learns useful feature extraction better than other techniques. [[paper](http://www.icml-2011.org/papers/455_icmlpaper.pdf)]

**More things to Explore** (not included on the test)

[Stacked Denoising Autoencoder](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

[Sequence-to-Sequence Autoencoder](https://machinelearningmastery.com/lstm-autoencoders/)

[Sliced-Wasserstein Auto-Encoder](https://arxiv.org/abs/1804.01947)

[Quantum Variational Autoencoder](https://arxiv.org/abs/1802.05779v2)

[Relational Autoencoder](https://arxiv.org/abs/1802.03145)

[Split Brain Autoencoder](https://arxiv.org/abs/1611.09842)

[Pretext tasks, self-supervised learning](https://arxiv.org/abs/1901.09005)

**Code Demos**

[Denoising Autoencoder in Keras](https://github.com/keras-team/keras/blob/master/examples/mnist_denoising_autoencoder.py)

[Variational Autoencoder in Tensorflow 2.0 alpha](https://www.tensorflow.org/alpha/tutorials/generative/cvae)