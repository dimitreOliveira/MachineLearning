### Support Vector Regression

Hello Wizards!

### Support Vector Machine

```
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimensional space, this hyperplane is a line dividing a plane into two parts wherein each class lay in either side.
```

To learn more about this please watch this video by our beloved director Siraj –  https://www.youtube.com/watch?v=g8D5YL6cOSE . (Not mandatory but will be really useful as SVM is a powerful toolbox.)

I have personally used one class SVM in a particular real-world use case for image classification and beat a neural network model’s accuracy and it is working well in production. http://isp.uv.es/papers/SPIE08_sssvdd.pdf

![](https://www.theschool.ai/wp-content/uploads/2019/02/i-support-vector-machines.jpg)

### Support Vector Regression.

Like always lets answer the first question **what** is SVR??

Support Vector Machine can likewise be utilized as a regression technique, keeping up all the primary highlights that portray the calculation (maximal margin). The Support Vector Regression (SVR) utilizes indistinguishable standards from the SVM for characterization, with just a couple of minor contrasts. As a matter of first importance, since yield is a real number it turns out to be hard to foresee the information at hand, which has limitless conceivable outcomes. On account of regression, a margin of resistance (epsilon) is set in estimate to the SVM which would have effectively requested for from the issue. Be that as it may, other than this reality, there is additionally a progressively confounded reason, the calculation is increasingly muddled consequently to be taken in thought. In any case, the principal thought is dependably the equivalent: to limit mistake, individualizing the hyperplane which augments the edge, remembering that piece of the blunder is endured.

Before we move forward let us discuss a few terminologies:

1. **Hyper Plane**:
    * In linear regression we used a best fitting straight line.
    * Remember from algebra, that the slope is the “m” in the formula **y = mx + b**.
    In the linear regression formula, the slope is the a in the equation **y’ = b + ax**.
    They are basically the same thing. So if you’re asked to find linear regression slope, all you need to do is find **b** in the same way that you would find **m**.
    * In SVR this is defined as a plane that will lead us to find continuous target value.
2. **Support vectors**: These are the data points which are closest to the hyperplane and help us relay boundary. The distance of the points is minimum or least.
3. **Boundary line**: There are two lines other than Hyper Plane which creates a margin. The support vectors can be on the Boundary lines or outside it. This boundary line separates the two classes.
4. **Kernel**: A map function that can convert lower dimensional data into a higher dimensional data.

![](https://www.theschool.ai/wp-content/uploads/2019/02/machine_learning.png)

Now **why** SVR?

In simple regression, our motive is to minimize the error rate. Instead in SVR, we try to fit the error within a certain threshold. The interesting part about SVR is that you can deploy a non-linear kernel. In this case, you end up creating a non-linear regression, i.e. fitting a curve rather than a line.

This process is based on the kernel trick and the representation of the solution/model in the dual rather than in the primal. That is, the model is represented as combinations of the training points rather than a function of the features and some weights. At the same time, the basic algorithm remains the same: the only real change in the process of going non-linear is the kernel function, which changes from a simple inner product to some nonlinear function.

![](https://www.theschool.ai/wp-content/uploads/2019/02/math-is-coming-no-point-resisting-it.jpg)

Now comes the interesting part! **How**?

### SVM

![](https://www.theschool.ai/wp-content/uploads/2019/02/37740.jpg)

**Support Vector Machine** can be applied not only to classification problems but also to the case of regression. Still, it contains all the main features that characterize maximum margin algorithm: a non-linear function is leaned by linear learning machine mapping into high dimensional kernel-induced feature space. The capacity of the system is controlled by parameters that do not depend on the dimensionality of feature space. In the same way as with the classification approach, there is motivation to seek and optimize the generalization bounds given for regression. They relied on defining the loss function that ignores errors, which are situated within a certain distance of the true value. This type of function is often called – epsilon intensive – loss function. The figure below shows an example of a one-dimensional linear regression function with – epsilon intensive – band. The variables measure the cost of the errors on the training points. These are zero for all points that are inside the band.

#### Linear SVR 

In the above graph, ε is the distance between hyperplane and the boundary line and denoted with a positive for upper and negative for lower.

Let us treat the hyperplane as a straight line y.

#### y = wx+b

Then the boundary lines can be given by

#### yi =  wxi + b + ε & yi = wxi + b – ε 

Thus coming in terms with the fact that for any linear hyperplane the equation that satisfy our SVR is: **ε ≤ y-wx-b ≤ + ε**

Thus the decision boundary is our Margin of tolerance i.e. we are going to take only those point that lies within this boundary or in simple terms that we are going to take only those points which have least error rate. Thus giving a better fitting model.

![](https://www.theschool.ai/wp-content/uploads/2019/02/i-am-currently-unsupervised-i-know-it-scares-me-too-6367215.png)

#### Non-Linear SVR

Using the right kernel functions we can transform the data into a higher dimensional feature space to make it possible to perform the linear separation.

#### Kernel functions

![](https://www.saedsayad.com/images/SVM_kernel_1.png)

I am giving a sample code for trying out SVR. Choose your dataset and try this out.

![](https://www.theschool.ai/wp-content/uploads/2019/02/Screen-Shot-2019-02-25-at-4.02.17-AM.png)

Happy learning!