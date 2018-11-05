import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# train is the training datat
# test is the test data
# y is the target variable for the train data
train = pd.DataFrame()
test = pd.DataFrame()
y = []

# split train data in 2 parts, training and validation.
training, valid, ytraining, yvalid = train_test_split(train, y, test_size=0.5)

# specify models
model1 = RandomForestRegressor()
model2 = LinearRegression()

# fir models
model1.fit(training, ytraining)
model2.fit(training, ytraining)

# make predictions for validation
preds1 = model1.predict(valid)
preds2 = model2.predict(valid)

# make predictions for test data
test_preds1 = model1.predict(test)
test_preds2 = model2.predict(test)

# Form a new dataset for valid and test via stacking the predictions
stacked_predictions = np.column_stack((preds1, preds2))
stacked_test_predictions = np.column_stack((test_preds1, test_preds2))

# specify meta model
meta_model = LinearRegression()

# fit meta model on stacked predictions
meta_model.fit(stacked_predictions, yvalid)

# make predictions on the stacked predictions of the test data
final_predictions = meta_model.predict(stacked_predictions)
