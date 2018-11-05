import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# train is the training datat
# test is the test data
# y is the target variable
train = pd.DataFrame()
test = pd.DataFrame()
y = []

model = RandomForestRegressor()
bags = 10
seed = 1

# create array object to hold bagged predictions
bagged_prediction = np.zeros(test.shape[0])

# loop for as many times as we want bags
for n in range(0, bags):
    model.set_params(random_state=seed + n)  # update seed
    model.fit(train, y)  # fit model
    preds = model.predict(test)  # predict on test data
    bagged_prediction += preds

# take average of predictions
bagged_prediction /= bags
