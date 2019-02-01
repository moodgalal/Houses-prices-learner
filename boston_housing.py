import numpy as np
import pandas as pd
import visuals as vs

df = pd.read_csv("housing.csv")
labels = df['MEDV']
features = df.drop('MEDV', 1)
print(df.shape)
print("Boston housing data contains {} data points with {} variables".format(*df.shape))

# Exploring Data
# Making descriptive statistics on the prices

maxVal = np.amax(labels)
minVal = np.amin(labels)
meanVal = np.mean(labels)
medianVal = np.median(labels)
sdVal = np.std(labels)
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minVal))
print("Maximum price: ${}".format(maxVal))
print("Mean price: ${:.2f}".format(meanVal))
print("Median price ${}".format(medianVal))
print("Standard deviation of prices: ${:.2f}".format(sdVal))

# Developing a model
# developing performance metric

from sklearn.metrics import r2_score


def performance_measurement(y_origin, y_predict):
    return r2_score(y_origin, y_predict)


# Goodness of fit (Assuming that we got these five entries from a model)
score = performance_measurement([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))


# Shuffle and split the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2, random_state=42)


# Analyzing the model

# Produce learning and complexity curves for varying training set sizes and maximum depths
vs.ModelLearning(features, labels)

# vs.ModelComplexity(x_train, y_train)

# Evaluating the model
# First we fit the data with the best parameters using grid search and cross validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ShuffleSplit


def model_fit(x, y):

    regressor = DecisionTreeRegressor(random_state=50)
    params = {'max_depth': range(1, 11)}
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.2)
    scoring_fnc = make_scorer(performance_measurement)

    grid = GridSearchCV(estimator=regressor, scoring=scoring_fnc, param_grid=params, cv=cv_sets)
    grid.fit(x, y)

    return grid.best_estimator_


clf = model_fit(x_train, y_train)
# Produce the best value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model. \n".format(clf.get_params()['max_depth']))

# Making predictions

# Produce a matrix for client data
client_data = [[5, 17, 15],  # Client 1
               [4, 32, 22],  # Client 2
               [8, 3, 12]]   # Client 3

prediction_result = clf.predict(client_data)
print(prediction_result)

for i, predicted_price in enumerate(prediction_result, 1):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i, predicted_price))
