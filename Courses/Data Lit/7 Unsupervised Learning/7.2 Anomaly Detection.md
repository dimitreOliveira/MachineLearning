## Anomaly Detection

* In this blog, I mainly cited
* https://www.datascience.com/blog/python-anomaly-detection
* Vegard Flovik's blog posting on towardsdatascience

![](https://www.theschool.ai/wp-content/uploads/2019/03/cat2.jpg)

## Introduction

* A technique used to identify unusual patterns that do not conform to expected behavior, called outliers.
* Similar to noise removal and novelty detection.
* Examples : intrusion detection in network, spotting a malignant tumor in an MRI scan, and fraud detection in credit card transactions.

### Types of anomalies
1. **Point anomalies**: A single instance of data is anomalous if it’s too far off from the rest. (ex: Detecting credit card fraud based on “amount spent.”)
2. **Contextual anomalies**: context specific, usually time-series data. (ex: Sending $100 on food every day during the holiday season is normal, but may be odd otherwise.)
3. **Collective anomalies**: A set of data instances collectively helps in detecting anomalies. (ex: Trying to copy data form a remote machine to a local host unexpectedly)
 
![](https://www.theschool.ai/wp-content/uploads/2019/03/2.jpeg)
###### It is helpful to understand datasets using Anomaly Detection techniques

## Anomaly Detection Techniques
### 1. Simple Statistical Approaches
* Computing the average across the data points, to find data point that deviates by a certain standard deviation from the mean.
* A moving average (rolling average) : to smooth short-term fluctuations and highlight long-term ones.
* Low pass filter : an n-period simple moving average. (ex: Kalman filter)
* Weak to the noise : the boundary between normal and abnormal behavior is often not precise.
* High variance : the threshold based on moving average may not always apply.
* The pattern is based on seasonality :  decomposing the data into multiple trends in order to identify the change in seasonality

### 2. Machine Learning-Based Approaches
1. **Density-Based**
    - K-nearest neighbor ( k-NN ): To classify data based on similarities in distance metrics such as Euclidean, Manhattan, Minkowski, or Hamming distance.
    - Relative density of data:  A local outlier factor (LOF). This concept is based on a distance metric called reachability distance.
2. **Clustering-Based**
    - K-means: It creates ‘k’ similar clusters of data points. Data instances that fall outside of these groups could potentially be marked as anomalies.
3. **Support Vector Machine-Based**
 - The extensions such as OneClassCVM can be used to identify anomalies as an unsupervised problems.
 - It learns a soft boundary in order to cluster the normal data instances using the training set.
 - Using the testing instance, it tunes itself to identify the abnormalities that fall outside the learned region.
4. **Autoencoder networks**
 - It learns a representation (encoding) for a set of data, typically for dimensionality reduction.
 - A reconstructing side: the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input.
 -It is trained on data representing the “normal” operating state.
 - By monitoring the re-construction error,  it can get an indication of the anomalies.

### 3. Multivariate statistical analysis
Visit [Vegard Flovik’s blog postng](https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7) for more details.

- Dimensionality reduction using principal component analysis: PCA
- Multivariate anomaly detection
- The Mahalanobis distance
 
![](https://www.theschool.ai/wp-content/uploads/2019/03/cat-1.jpg)

## Code
#### Simple Detection Solution using a Low-Pass Filter

It runs under Python 3 and Jupyter notebook. You can try this using colab here:

[Google colab notebook](https://www.theschool.ai/wp-content/uploads/2019/03/colab.png)


or visit [github link](https://github.com/decoderkurt/DataLit_week7_anomaly_detection/blob/master/DataLit_week7_anomaly_detection.ipynb)

### 1.  Download sample dataset
```
!wget -c -b http://www-personal.umich.edu/~mejn/cp/data/sunspots.txt
```

### 2. Dependencies
```
import sys
from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
style.use('fivethirtyeight')
%matplotlib inline
```

### 3. Datasets
```
# 1. Download sunspot dataset and upload the same to dataset directory
#    Load the sunspot dataset as an Array
!mkdir -p dataset
!wget -c -b http://www-personal.umich.edu/~mejn/cp/data/sunspots.txt -P dataset
data = loadtxt("dataset/sunspots.txt", float)
```
```
# 2. View the data as a table
data_as_frame = pd.DataFrame(data, columns=['Months', 'SunSpots'])
data_as_frame.head()
```

![](https://www.theschool.ai/wp-content/uploads/2019/03/00.png)

### 4. Define functions
```
# 3. Lets define some use-case specific UDF(User Defined Functions)

def moving_average(data, window_size):
    """Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
        data (pandas.Series): independent variable
        window_size (int): rolling window size
    Returns:
        ndarray of linear convolution
    References:
        [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
        [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def explain_anomalies(y, window_size, sigma=1.0):
    """Helps in exploring the anamolies using stationary standard deviation
    Args:
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation
    Returns:
          a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
          containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size).tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for
                                                       index, y_i, avg_i in zip(count(), y, avg)
                                                       if(y_i > avg_i + (sigma * std)) | (y_i < avg_i - (sigma * std))])}


def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
    """Helps in exploring the anamolies using rolling standard deviation
    Args:
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation
    Returns:
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                            testing_std_as_df.ix[window_size - 1]).round(3).iloc[:, 0].tolist()
    std = np.std(residual)
    return {'stationary standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i)
                                                       for index, y_i, avg_i, rs_i in zip(count(),y, avg_list,rolling_std)
                                                       if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}


# This function is repsonsible for displaying how the function performs on the given dataset.
def plot_results(x, y, window_size, sigma_value=1, title_for_plot="",
                 text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
    """Helps in generating the plot and flagging the anamolies.
        Supports both moving and stationary standard deviation. Use the 'applying_rolling_std' to switch
        between the two.
    Args:
        x (pandas.Series): dependent variable
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma_value (int): value for standard deviation
        title_for_plot (str): title
        text_xlabel (str): label for annotating the X Axis
        text_ylabel (str): label for annotatin the Y Axis
        applying_rolling_std (boolean): True/False for using rolling vs stationary standard deviation
    """
    plt.figure(figsize=(15, 8))
    plt.title(title_for_plot)
    plt.plot(x, y, "k.")

    y_av = moving_average(y, window_size)
    plt.plot(x, y_av, color='green')
    plt.xlim(0, 1000)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)

    # Query for the anomalies and plot the same
    events = {}
    if applying_rolling_std:
        events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
    else:
        events = explain_anomalies(y, window_size=window_size, sigma=sigma_value)

    x_anomaly = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
    y_anomaly = np.fromiter(events['anomalies_dict'].values(), dtype=float,
                            count=len(events['anomalies_dict']))
    plt.plot(x_anomaly, y_anomaly, "r*", markersize=12)

    # add grid and lines and enable the plot
    plt.grid(True)
    plt.show()
```

### 5. Plot
```
# 4. Lets play with the functions
x = data_as_frame['Months']
Y = data_as_frame['SunSpots']

# plot the results
plot_results(x, y=Y, window_size=10, text_xlabel="Months", sigma_value=3,
             text_ylabel="No. of Sun spots")
events = explain_anomalies(Y, window_size=5, sigma=3)

# Display the anomaly dict
print("Information about the anomalies model:{}".format(events))
```

![](https://www.theschool.ai/wp-content/uploads/2019/03/11.png)

### 6. Other examples (using a random dataset)
```
# Convenience function to add noise
def noise(yval):
    """ Helper function to generate random points """
    np.random.seed(0)
    return 0.2 * np.asarray(yval) * np.random.normal(size=len(yval))


# Generate a random dataset
def generate_random_dataset(size_of_array=1000, random_state=0):
    """ Helps in generating a random dataset which has a normal distribution
    Args:
    -----
        size_of_array (int): number of data points
        random_state (int): to initialize a random state
    Returns:
    --------
        a list of data points for dependent variable, pandas.Series of independent variable
    """
    np.random.seed(random_state)
    y = np.random.normal(0, 0.5, size_of_array)
    x = range(0, size_of_array)
    y_new = [y_i + index ** ((size_of_array - index) / size_of_array) + noise(y)
             for index, y_i in zip(count(), y)]
    return pd.Series(x), pd.Series(y)


# Lets play
x1, y1 = generate_random_dataset()
# Using stationary standard deviation over a continuous sample replicating
plot_results(x1, y1, window_size=12, title_for_plot="Statinoary Standard Deviation",
                    sigma_value=2, text_xlabel="Time in Days", text_ylabel="Value in $")

# using rolling standard deviation for
x1, y1 = generate_random_dataset()
plot_results(x1, y1, window_size=50, title_for_plot="Using rolling standard deviation",
             sigma_value=2, text_xlabel="Time in Days", text_ylabel="Value in $", applying_rolling_std=True)
```

![](https://www.theschool.ai/wp-content/uploads/2019/03/22.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/33.png)
![](https://www.theschool.ai/wp-content/uploads/2019/03/rabbit.jpg)

## Conclusion
We looked at the techniques of anomaly detection  including simple statistical methods, how to use machine learning, and actual codes.
Since it can be used in a variety of fields, including real world problem solving and business utilization, I recommend you learn it well and study it in depth.

- What is anomaly detection
- Types of anomalies.
- 3 Anomaly Detection Techniques
    1. Simple Statistical Approaches
    2. Machine Learning-Based Approaches
    3. Multivariate statistical analysis
- Python code using a low pass filter