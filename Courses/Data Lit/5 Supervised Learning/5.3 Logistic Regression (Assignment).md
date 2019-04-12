### Logistic Regression (Assignment)

Send your submission to gradedhomeworkassignments@gmail.com for a grade!

Last week we learned about logistic regression, one of the most important and fundamental machine learning (ML) algorithms. It is time for you to execute what you have learned in the class to really implement the logistic regression model. Instead of covering many algorithms quickly at a high-level and using ML algorithm libraries (e.g. [Scikit-learn](http://scikit-learn.org/stable/)), this course aims to explain a few important ML algorithms in-depth. It is meant to prepare you to be a researcher/scientist — the very top class in the [Machine Learning Skills Pyramid](http://socialmedia-class.org/assets/img/ml_pyramid.jpg), who can not only apply algorithms but create new algorithms!

I am going to separate the assignment into three parts.

1. Choose a dataset/problem of your choice or use the followingUS Adult Census data relating income to social factors such as Age, Education, race etc.The Us Adult income dataset was extracted by Barry Becker from the 1994 US Census Database. The data set consists of anonymous information such as occupation, age, native country, race, capital gain, capital loss, education, work class and more. Each row is labelled as either having a salary greater than “>50K” or “<=50K”.
This Data set is split into two CSV files, named adult-training.txt and adult-test.txt.

The goal here is to train a binary classifier on the training dataset to predict the column income_bracket which has two possible values “>50K” and “<=50K” and evaluate the accuracy of the classifier with the test dataset.

Note that the dataset is made up of categorical and continuous features. It also contains missing values The categorical columns are: workclass, education, marital_status, occupation, relationship, race, gender, native_country

https://www.kaggle.com/johnolafenwa/us-census-data

#### Prepare the data.

1. Apply sklearn or your favorite ML library’s inbuilt logistic regression model for prediction.
2. In this assignment, you are expected to implement logistic regression and get a good understanding of the key components of logistic regression:
    * hypothesis function
    * cost function
    * decision boundary
    * gradient descent algorithm

Finally, compare the result you got from built-in model versus your own implementation on various metrics.

Like always you can submit your GitHub link for jupyter notebook or google colab explaining your steps.

All the best.

If you need help use this example: https://www.kaggle.com/kost13/us-income-logistic-regression/notebook