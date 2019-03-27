### Generative Adversarial Networks

In this blog, I mainly cited

- https://towardsdatascience.com/generative-adversarial-networks-explained-34472718707a
- https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900
- http://kvfrans.com/generative-adversial-networks-explained/
- [Devnag's GAN in 50 lines of code](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f)

![](https://www.theschool.ai/wp-content/uploads/2019/03/catchme.jpg)
![](https://www.theschool.ai/wp-content/uploads/2019/03/gan-1024x239.png)

## Introduction

It’s finally time to study GAN. It is such a popular subject that I recommend you to study the further reading at the end of the blog posting.

- Generative Adversarial Networks takes up a game-theoretic approach.
- The network learns to generate from a training distribution through a 2-player game.
- The two entities are Generator and Discriminator. These two adversaries are in constant battle throughout the training process. Since an adversarial learning method is adopted, we need not care about approximating intractable density functions.
- The main focus for GAN (Generative Adversarial Networks) is to generate data from scratch, mostly images but other domains including music have been done.

![](https://www.theschool.ai/wp-content/uploads/2019/03/zebra.gif)
##### Right side image created by CycleGAN [Source](https://junyanz.github.io/CycleGAN/)

- It attempts to train an image generator by simultaneously training a discriminator to challenge it to improve.

## Applications

GAN can be used almost everywhere! Let’s see some of them quickly.

![](https://www.theschool.ai/wp-content/uploads/2019/03/ga6-768x409.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga7-768x448.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga8-768x493.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga10-768x302.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga11-768x293.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga12-768x303.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga13-768x181.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga1-600x640.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga2-600x368.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga3-768x116.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga4-600x548.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/ga5-600x380.jpeg)

## Intuitive example

Let’s think of a back-and-forth situation between a bank and a money counterfeiter.

- At the beginning, the fakes are easy to spot. However, as the counterfeiter keeps trying different kinds of techniques, some may get past the check. The counterfeiter then can improve his fakes towards the areas that got past the bank’s security checks.
- But the bank doesn’t give up. It also keeps learning how to tell the fakes apart from real money.
- After a long period of back-and-forth, the competition has led the money counterfeiter to create perfect replicas.

## Generator and discriminator

![](https://www.theschool.ai/wp-content/uploads/2019/03/gan22-1024x360.png)

GAN composes of two deep networks, the generator, and the discriminator.

- The generator tries to fool the discriminator, while it tries not to be fooled.
- The counterfeiter is known as the generative network, and is a special kind of convolutional network that uses transpose convolutions, (known as a deconvolutional network.)

![](https://www.theschool.ai/wp-content/uploads/2019/03/gen1-1024x440.png)
##### How images are generated from deconvolutional layers. [[source]](https://openai.com/blog/generative-models/)

#### How it works

- A generator is used to generate real-looking images and the discriminator’s job is to identify which one is a fake.
- The input, random noise can be a Gaussian distribution and values can be sampled from this distribution and fed into the generator network and an image is generated.
- This generated image is compared with a real image by the discriminator and it tries to identify if the given image is fake or real.

![](https://www.theschool.ai/wp-content/uploads/2019/03/gan2-768x335.png)

### Tips
- If the generator is not good enough,
    - it will never be able to fool the discriminator
    - the model will never converge.
- If the discriminator is bad,
    - then images which make no sense will also be classified as real
    - the model never trains and in turn you never produces the desired output.

## Objective Function

### The Generator

![](https://www.theschool.ai/wp-content/uploads/2019/03/gd-600x88.png)
##### Gradient Descent on Generator

- It tries to minimize the objective function.
- It performs gradient descent on the function.
- Optimizing the generator objective function does not work so well
    - Because when the sample is generated, it is likely to be classified as fake.
    - The gradients turn out to be relatively flat.
- Therefore, the generator objective function is changed as below.

![](https://www.theschool.ai/wp-content/uploads/2019/03/gd2-600x108.png)
##### New Generator Objective function (gradient ascent on generator)

- Instead of minimizing the likelihood of discriminator being correct.
- We maximize the likelihood of discriminator being wrong.
- We perform gradient ascent on generator according to this objective function.

### The discriminator

![](https://www.theschool.ai/wp-content/uploads/2019/03/ga-768x70.png)
##### Gradient Ascent on Discriminator

- It tries to maximize the objective function.
- It performs gradient ascent on the function.

By alternating between gradient ascent and descent, the network can be trained.

![](https://www.theschool.ai/wp-content/uploads/2019/03/gan3-1024x145.png)

#### Just 50 lines of Code (PyTorch)
You can see full codes here : https://github.com/devnag/pytorch-generative-adversarial-networks

![](https://www.theschool.ai/wp-content/uploads/2019/03/code-768x370.png)

There are really only 5 components to think about:

- **R**: The original, genuine data set
- **I**: The random noise that goes into the generator as a source of entropy
- **G**: The generator which tries to copy/mimic the original data set
- **D**: The discriminator which tries to tell apart **G**’s output from **R**
- The actual ‘training’ loop where we teach **G** to trick **D** and **D** to beware **G**.

**1.) R (Real Data)**: A bell curve.

- It takes a mean and a standard deviation.
- It returns a function which provides the right shape of sample data from a Gaussian with those parameters.
- We’ll use a **mean of 4.0** and a **standard deviation of 1.25**.

![](https://www.theschool.ai/wp-content/uploads/2019/03/c1.png)

**2.) I (Input)**: random

- To make our job a little bit harder, we will use a uniform distribution rather than a normal one.
- **G** can’t simply shift/scale the input to copy R, but has to reshape the data in a non-linear way.

![](https://www.theschool.ai/wp-content/uploads/2019/03/c2.png)

**3.) G(Generator)**: A standard feed-forward graph.

- Two hidden layers, three linear maps.
- We’re using a hyperbolic tangent activation function.
- It is going to get the uniformly distributed data samples from **I**.
- It mimics the normally distributed samples from **R** without ever seeing it.

![](https://www.theschool.ai/wp-content/uploads/2019/03/c3-600x300.png)

**4.) D (Discriminator)**: A feed-forward graph with two hidden layers and three linear maps. (like **G**)

- The activation function here is a sigmoid.
- It’s going to get samples from either **R** or **G**.
- It will output a single scalar between 0 and 1, interpreted as ‘fake’ vs. ‘real’.

![](https://www.theschool.ai/wp-content/uploads/2019/03/c4-768x314.png)

**5.)** The training loop alternates between two modes

- Training **D** on real data vs. fake data, with accurate labels.
- Training **G** to fool **D**, with inaccurate labels.

![](https://www.theschool.ai/wp-content/uploads/2019/03/c5.png)

## Disadvantages

- GANs are more unstable to train because you have to train two networks from a single backpropagation.
- Therefore choosing the right objectives can make a big difference.
- We cannot perform any inference queries with GANs

There are several failure modes that we need to avoid:

- **Discriminator losses approach zero**: this leaves practically no gradients for the generator’s optimizer.
- **Discriminator losses rise unbounded on generated images**: similarly, this leaves practically no gradient for the discriminator to improve, and the generator’s training stalls, too, since the gradients it’s reading suggest that it has achieved perfect performance.
- **Divergent discriminator accuracy**: the discriminator learns a shortcut by either classifying everything as real or everything as generated. You can detect this by checking the discriminator’s losses on generated images against the discriminator’s losses on real images.

## Conclusion

- GAN uses game-theoretic approach between two players. (The generator and the discriminator)
- The generator tries to minimize the objective function while the discriminator tries to maximize it.
- It is still hard to train GAN because you have to train two networks from a single backpropagation.
- After finishing the learning, ‘realistic fakes’ are made from GAN.
- The fakes are what we need in the end. Therefore, the generator is the key. (we can remove the discriminator from GAN after the learning.)

![](https://www.theschool.ai/wp-content/uploads/2019/03/gan_summary-600x606.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/catchme2.jpg)

## Further reading

![](https://www.theschool.ai/wp-content/uploads/2019/03/gans.png)

- https://colab.research.google.com/github/llSourcell/Generative_Adversarial_networks_LIVE/blob/master/EZGAN.ipynb
EZGAN code by Siraj Raval on colab
- https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb
BIGGAN code by TF on colab
- https://medium.com/@jonathan_hui/gan-gan-series-2d279f906e7b
Great summary of GAN series.
- https://towardsdatascience.com/gan-objective-functions-gans-and-their-variations-ad77340bce3c
Great summary of GAN variations.