### Gradient descent

Hello Wizards! Today we are going to see the most important algorithm of deep learning.

![](https://www.theschool.ai/wp-content/uploads/2019/03/Picture1-1.png)

Like always let’s start with our first question what is gradient descent?

Gradient descent is an algorithm that helps to minimize some function by iteratively descending in the direction of the fastest descent as defined by the negative of the gradient.

![](https://www.theschool.ai/wp-content/uploads/2019/03/mount.png)

The pinnacle of the mountain is the starting point and reaching the base is the destination. There can be many ways that can lead us to the bottom of the mountain but the fastest way to reach the bottom will be in the steepest direction. We can take many small steps and change the weights in steps that reduce the distance to the bottom. So here the problem is the mountain and the process to solve the problem is to follow the tiny steps in the correct direction that will get us to the solution.

Now let’s move to the second question How?

Following the analogy above, we can find the tiny steps in the right direction by calculation the gradient of the squared error. Gradient means the rate of change or slope of a function.

How can we calculate the rate of change?

To go a bit deeper let’s run over calculus. A derivative of a function f(x) gives us another function f’(x) that gives us the slope of  f(x) at point x. we can use this to find the gradient at any point in the error function we choose which depends on the input weights.

![](https://www.theschool.ai/wp-content/uploads/2019/03/Screen-Shot-2019-03-25-at-4.49.36-PM.png)

This picture symbolizes the top to bottom view of the mountain and the blue arrow show the steps to the solution. Here the gradient is a vector that contains the direction of the steepest step along with the distance.

[Here](https://developers.google.com/machine-learning/crash-course/fitter/graph) is a cool explanation from [the Machine Learning crash course](https://developers.google.com/machine-learning) from Google, where you can visually see the effects of the learning rate. [Link here](https://developers.google.com/machine-learning/crash-course/fitter/graph).

Here the agent only knows two things: the gradient ( for that position, or parameters) and the distance of the step to take ( learning rate). With the help of these two, we can update the current value of each parameter. With the new parameters the gradient is recalculated and the process is repeated until we reach the convergence or local minima. **Convergence** is a name given to the situation where the loss function does not improve significantly, and we are stuck in a point near to the minima.

![](https://www.theschool.ai/wp-content/uploads/2019/03/Screen-Shot-2019-03-25-at-5.12.52-PM.png)

To learn more about Gradient descent use these links:

https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html