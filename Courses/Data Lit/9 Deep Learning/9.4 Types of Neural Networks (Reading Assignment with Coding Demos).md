### Types of Neural Networks (Reading Assignment with Coding Demos)

- Data-Lit Week 9
- Carson Bentley       March 25, 2019
- Types of Neural networks
- Reading Assignment with Coding Demos
 

#### Introduction

Hello again wizards, and welcome to week 9. I have collect, tested, and polished a series of neural network coding demos with a short explanation and links to background material for each. I hope that these materials are useful to you in your quest of knowledge.

#### Perceptron

The perceptron is the most simple form of neural network, composed of just one layer. It can be used to solve a binary classification task. Notably, if we attempt to solve an XOR type classification problem, the perceptron fails to capture a meaningful representation. The term XOR just means exclusive or: A or B but not both. If you picture this as a checkerboard, it’s easy to see that there is no single straight line that you can draw to separate the two colors.

[The Perceptron – A Receiving and Recognizing Automaton](https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf) (1957)

[Perceptron training handout](https://colab.research.google.com/drive/1dUzgdITJpNXJIEsoqRtEdR7wA9Biihg9)

Colab notebook by Professor [Fabio A. González O](http://dis.unal.edu.co/~fgonza/).


#### Multilayer Perceptron / Feed Forward Neural Network

A multilayer perceptron is essentially the same thing as a feed forward neural network, the form of algorithm that we first discussed in week 5. The layers are commonly described as being ‘densely connected’ (dense layers in keras). The key insight that allows for efficient multiple layers is the backpropagation algorithm, introduced by Geoffrey Hinton in 1986.

[Learning Internal Representations by Error Propagation](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf) (1986)

[Train your first neural network: basic classification](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/basic_classification.ipynb)

Colab notebook by the Tensorflow team at Google


#### Neocognitron / Convolutional Neural Network

The principal ancestor of modern convolutional neural networks is the ‘neocognitron’, invented by Kunihiko Fukushima in 1980. This work introduced convolutional layers as well as downsampling, the two main components that make up the modern CNN. Yann LeCun’s work with handwritten zip codes is also notable, as this brought together the neocognitron and backpropagation.

[Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) (1980)

[Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) (1989)

[Convolutional Neural Networks](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/images/intro_to_cnns.ipynb)

Colab notebook by the Tensorflow team at Google


#### Autoencoders

We just discussed autoencoders last weeks, so it should still be fairly fresh in your mind. As a reminder, the basic idea is to train a network to reproduce the input signal.

[Modular Learning in Neural Networks](https://www.aaai.org/Papers/AAAI/1987/AAAI87-050.pdf) (1987)

[Autoencoder Denoising Demo](https://colab.research.google.com/github/aztecman/DataLit/blob/master/Denoising_Autoencoder_Demo.ipynb)

Colab notebook based on a Keras team example


#### Generative Adversarial Networks

The idea behind GANs is to train a network to produce fake pieces of data, while another network simultaneously trains to discern the fakes from the originals. Ian Goodfellow is the main researcher behind this architecture, which has enjoyed great popularity in recent years.

[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (2014)

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) (2015)

[Deep Convolutional Generative Adversarial Network](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/dcgan.ipynb)

Colab notebook by the Tensorflow team at Google


#### Recurrent Neural Networks , Long Short Term Memory (units)

RNNs emerged in the eighties as a way to apply neural nets to sequential data. The domain of natural language processing (NLP) benefits greatly from this innovation. LSTMs were developed in the late 90s by Juergen Schmidhuber as a refinement of this architecture. The ‘long short-term memory’ unit introduced the capacity to forget. It should be noted that the term RNN is often used interchangeably with LSTM.

[Learning State Space Trajectories in Recurrent Neural Networks](http://www.bcl.hamilton.ie/~barak/papers/NC-dynets-89.pdf) (1989)

[Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (1997)

[Text classification with an RNN](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/sequences/text_classification_rnn.ipynb)

Colab notebook by the Tensorflow team at Google


#### Gated Recurrent Units

GRUs, similar to RNNs and LSTMs, are a general purpose unit for neural networks dealing in sequential data. They were developed in particular as a way to handle language translation. There [is some evidence](https://arxiv.org/abs/1412.3555) that they are roughly equivalent in effectiveness to LSTMs.

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) (2014)

[Sentiment Analysis with GRUs](https://colab.research.google.com/github/djcordhose/ai/blob/master/notebooks/tensorflow/sentiment-gru.ipynb)

Colab notebook by Oliver Zeigermann based on prior work by the Keras team


#### Graph Neural Networks, Graph Convolutional Networks

Graph neural networks are specialized to handle information that takes the form of a graph. A graph is any data which can be represented by nodes connected with edges. Graph based data includes social networks, genealogy trees, and 3D models. The more recent Graph convolutional network (GCN) borrows the concept of a ‘receptive field’ from CNNs.

[Graph Neural Networks for Ranking Web Pages](http://delab.csd.auth.gr/~dimitris/courses/ir_spring06/page_rank_computing/01517930.pdf) (2005)

[The graph neural network model](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1015.7227&rep=rep1&type=pdf) (2009)

[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (2016)

[Graph Convolutional Network](https://colab.research.google.com/gist/aztecman/b94bc5b5de1d58dfec38c9f5ab5ef47b/1_gcn.ipynb)

Colab notebook by Qi Huang, taken from [DGL library docs](https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/1_gcn.html)


#### Capsule Networks

Introduced by Geoffrey Hinton in 2017, capsule networks are a highly effective alternative to convolutional neural networks for image data. One of the main ideas behind them is to use inverse graphics to construct an internal representation for our network.

[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) (2017)

[Capsule Network](https://colab.research.google.com/github/TheAILearner/Capsule-Network/blob/master/Capsule%20Network.ipynb)

Colab notebook by [Kang & Atul](https://theailearner.com/2019/01/21/capsule-networks/)


#### Neural Ordinary Differential Equations

Last, but not least, NODEs are a radically different approach to the others we have seen so far. The main idea is to replace the series of discrete transformations created by hidden units (layers) with a continuous transformation based on a vector field.

[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (2018)

[PyTorch Implementation of Differentiable ODE Solvers: Demo](https://colab.research.google.com/github/aztecman/DataLit/blob/master/NODEs_DEMO.ipynb)

Colab notebook based on [Ricky Chen’s Github](https://github.com/rtqichen/torchdiffeq)