### Homework Assignment (Unsupervised Learning)

## Clustering Assignment
 
##### Assignment instructions:

1. Use [Google Co-lab](https://colab.research.google.com/) or Jupyter.
2. Save your workbook to a Github repository
3. Comment the link below.
 

1. List the types of clustering and give one real world example that you could come up with for each and justify your choice. 
    - STL-10: http://cs.stanford.edu/˜acoates/stl10/ 
    - Labeled Faces in the Wild: http://vis-www.cs.umass.edu/lfw/  Choose choice of your library for K-means clustering. Go out and grab an image data set like:
        - CIFAR-10 or CIFAR-100: http://www.cs.toronto.edu/˜kriz/cifar.html 
        - MNIST Handwritten Digits: http://yann.lecun.com/exdb/mnist/ 
        - Small NORB (toys): http://www.cs.nyu.edu/˜ylclab/data/norb-v1.0-small/ 
        - Street View Housing Numbers: http://ufldl.stanford.edu/housenumbers/
         
2. Figure out how to load it into your environment and turn it into a set of vectors. Run K-Means on it for a few different K and show some results from the fit. What do the mean images look like? What are some representative images from each of the clusters? Are the results wildly different for different restarts and/or different K? Plot the K-Means objective function (distortion measure) as a function of iteration and verify that it never increases.   
3. Implement K-Means clustering from scratch and apply the choice of your dataset you used above and compare the results.  