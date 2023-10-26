import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, Ridge, RidgeCV
from sklearn.ensemble import IsolationForest
from sklearn import model_selection

# get current directory
cur_dir = os.getcwd()
data_dir = cur_dir + '/data'
output_dir = cur_dir + '/output'

# used to randomize the indexes
random_indices = random.sample(range(0, 100), 100)

# open the files from data1 folder that contain the np arrays
X = np.load(data_dir + '/Xtrain_Regression2.npy')
Y = np.load(data_dir + '/Ytrain_Regression2.npy')
#VER ISTO
X_test = np.load(data_dir + '/Xtest_Regression2.npy')

#########################################
#########   ISOLATION FOREST   ##########
#########################################

print("\nISOLATION FOREST:\n")

# Create isolation forest model
iforest = IsolationForest()

# Code for the cross-validation of the isolation foret hyperparameters
'''
param_grid = {'max_samples': [30, 50], 
              'contamination': [0.20], 
              'max_features': [2]}

grid_search = model_selection.GridSearchCV(iforest, 
                                           param_grid,
                                           scoring="neg_mean_squared_error", 
                                           refit=True,
                                           cv=10, 
                                           return_train_score=True)


best_model = grid_search.fit(X, Y)

print('Optimum parameters', best_model.best_params_)
'''

#Optimum parameters {'contamination': 0.2, 'max_features': 2, 'max_samples': 30}

# Once optimum parameters are identified, use the isolation forest model with those parameters
iforest = IsolationForest(contamination=0.2, 
                          max_features=2,
                          max_samples=30, 
                          n_estimators=1000)

# Append output into input matrix
X_join_Y = X

X_join_Y = np.append(X, Y, axis = 1)

# Predict outliers
# Vector identifyng the outliers (is equal to -1 in points identified as outliers)
outlier_flags = iforest.fit_predict(X_join_Y)
# Vector with outlier score for each point
outlier_scores = iforest.decision_function(X_join_Y)

# Remove the outliers from the data
x_filtered, y_filtered = X[(outlier_flags != -1), :], Y[(outlier_flags != -1)]

# Create linear regressor model
lnlg = LinearRegression()

# Fit and predict model and calculate sse with the outliers
lnlg.fit(X, Y)
y_pred = lnlg.predict(x_filtered)
sse = np.sum((np.subtract(y_filtered, y_pred))**2)

# Fit and predict model and calculate sse without the outliers
lnlg.fit(x_filtered, y_filtered)
y_pred_filtered = lnlg.predict(x_filtered)
sse_filtered = np.sum((np.subtract(y_filtered, y_pred_filtered))**2)

print("SSE without outlier filtering:" + str(sse))
print("SSE with outlier filtering:" + str(sse_filtered))

#########################################
#########   HUBER REGRESSOR   ###########
#########################################

print("\nHUBER REGRESSOR:\n")

# Create huber model
huber = HuberRegressor()
# Create RANSAC model
ransac = RANSACRegressor()
# Vector with multiple epsilon values to cross-validate
epsilon_values = np.arange(1, 5, 0.05)

# Vector to store SSE for each epsilon value
sse = []

# For each epsilon value
for k, epsilon in enumerate(epsilon_values):

    # Fit Huber model
    huber.fit(X,Y.ravel())    

    # Fit RANSAC model
    ransac.fit(X, Y)

    # Create inlier mask to identify inliears and outliers
    inlier_mask = ransac.inlier_mask_

    # Remove outliers
    x_filtered, y_filtered = X[(inlier_mask != False), :], Y[(inlier_mask != False)]

    # Predict ouput values for the training set
    y_pred = huber.predict(x_filtered)

    # Insert SSE into the SSE vector
    sse.append( (np.sum( np.subtract(y_filtered.ravel(), y_pred))**2))

# Get optimum epsilon (epsilon whichc results in the lowest SSE)
optimum_epsilon = epsilon_values[np.argmin(sse)]
# Get minimum SSE
sse_min = np.min(sse)
# Mask to identify minimum point (epsilon, SSE)
mask = np.array(sse) == sse_min
# Set minimum to red for the plot
color = np.where(mask, 'red', 'blue')

print("Optimum epsilon: " + str(optimum_epsilon))
print("SSE: " + str(sse_min))

# plot graph for the SSE for different epsilon values
plt.scatter(epsilon_values, sse, color=color)
plt.title("Comparison of HuberRegressor SSE for different epsilon values")
plt.xlabel("epsilon")
plt.ylabel("sse")
plt.legend(loc=0)
plt.show()

#########################################
#########   RANSAC ALGORYTHM   ##########
#########################################

print("\nRANSAC ALGORYTHM WITH RIDGE REGRESSION:\n")

ransac = RANSACRegressor()

ransac.fit(X, Y)

inlier_mask = ransac.inlier_mask_

x_filtered, y_filtered = X[(inlier_mask != False), :], Y[(inlier_mask != False)]

# Use code from first submission for the Ridge regression with the optimum alpha
alphas_ridge = np.logspace(-3, 3)
k = 10
kf = KFold(n_splits=k)
kf.get_n_splits(x_filtered)

# Using sklearn K-fold Cross validation algorithm for Ridge regression model
print("\nRIDGE REGRESSION MODEL:\n")
SSE_ridge = []
Alpha_ridge = []
i = 1
for train_index, test_index in kf.split(x_filtered):
    Xtrain, Xtest = x_filtered[train_index], x_filtered[test_index]
    Ytrain, Ytest = y_filtered[train_index], y_filtered[test_index]
    ridgeAlpha = RidgeCV(alphas=alphas_ridge).fit(Xtrain, Ytrain)
    ridge = Ridge(alpha=ridgeAlpha.alpha_, random_state=0)
    ridge.fit(Xtrain, Ytrain)
    pred_values = ridge.predict(Xtest)
    sse_ridge = np.sum(np.subtract(Ytest, pred_values) ** 2)
    SSE_ridge.append(sse_ridge)
    Alpha_ridge.append(ridgeAlpha.alpha_)
    i += 1
print("Mean of SSE: ", np.mean(SSE_ridge))
print("Standard deviation of SSE: ", np.std(SSE_ridge))
print("Alphas: ", Alpha_ridge)
print("Mean of Alphas: ", np.mean(Alpha_ridge))
print("Standard deviation of Alphas: ", np.std(Alpha_ridge))

# Using sklearn K-fold Cross validation algorithm for best parameter of Ridge Regression
print("\nRIDGE REGRESSION MODEL FINAL VALUES:\n")
SSE_ridge = []
alpha = np.mean(Alpha_ridge)
ridge = Ridge(alpha=alpha, random_state=0)
i = 1
for train_index, test_index in kf.split(x_filtered):
    Xtrain, Xtest = x_filtered[train_index], x_filtered[test_index]
    Ytrain, Ytest = y_filtered[train_index], y_filtered[test_index]
    ridge.fit(Xtrain, Ytrain)
    pred_values = ridge.predict(Xtest)
    sse_ridge = np.sum(np.subtract(Ytest, pred_values) ** 2)
    SSE_ridge.append(sse_ridge)
    print("Fold: %2d, Sum of Squared Errors with Sklearn: %.3f " % (i, sse_ridge))
    i += 1
print("Mean of SSE: ", np.mean(SSE_ridge))
print("Standard deviation of SSE: ", np.std(SSE_ridge))

# Predict output for the test set
Y_test = huber.predict(X_test)
# Save predicted output
np.save(output_dir + "/Ytest_Regression2", Y_test)

y2 = np.load(output_dir + '/Ytest_Regression2.npy')
print(y2)