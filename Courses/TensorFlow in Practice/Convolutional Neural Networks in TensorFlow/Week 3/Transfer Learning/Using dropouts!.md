Another useful tool to explore at this point is the Dropout.

The idea behind Dropouts is that they remove a random number of neurons in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit!

Check out Andrew's terrific video explaining dropouts here: https://www.youtube.com/watch?v=ARq74QuavAo

