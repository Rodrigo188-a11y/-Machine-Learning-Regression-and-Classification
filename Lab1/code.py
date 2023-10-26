import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, Lasso
from sklearn.utils import column_or_1d


def plot_linear():
    data_linalg = {'Folds': Folds, 'SSE': SSE_linalg}
    df_linalg = pd.DataFrame(data=data_linalg)
    sns.barplot(data=df_linalg, x='Folds', y='SSE', errorbar="sd")
    # giving title to the plot
    plt.title('SSE LINEAR REGRESSION');
    # function to show plot
    plt.show()
    plt.close()


def plot_ridge_sse():
    data_ridge = {'Folds': Folds, 'SSE': SSE_ridge}
    df_ridge = pd.DataFrame(data=data_ridge)
    sns.barplot(data=df_ridge, x='Folds', y='SSE', errorbar="sd")
    # giving title to the plot
    plt.title('SSE RIDGE REGRESSION');
    # function to show plot
    plt.show()
    plt.close()


def plot_ridge_alpha():
    data1_ridge = {'Folds': Folds, 'Alphas': Alpha_ridge}
    df1_ridge = pd.DataFrame(data=data1_ridge)
    sns.barplot(data=df1_ridge, x='Folds', y='Alphas', errorbar="sd")
    # giving title to the plot
    plt.title('Alphas RIDGE REGRESSION');
    # function to show plot
    plt.show()
    plt.close()


####  Data Reading from files and test data statistics  ###############
# get current directory
cur_dir = os.getcwd()
data_dir = cur_dir + '/data'
output_dir = cur_dir + '/output'

# open the files from data1 folder that contain the np arrays
X_train = np.load(data_dir + '/Xtrain_Regression1.npy')
# Calculate mean and standard deviaton of training data
x_mean = np.mean(X_train, axis=0)
x_std_dev = np.std(X_train, axis=0)
# Standardize training set
X = np.divide(np.subtract(X_train, x_mean), x_std_dev)
Y = np.load(data_dir + '/Ytrain_Regression1.npy')
# Standardize test set
Xtest_geral = np.load(data_dir + '/Xtest_Regression1.npy')
X_test = np.divide(np.subtract(Xtest_geral, x_mean), x_std_dev)


# Method to evaluate if data is gaussian
stat, p = shapiro(Y)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

pyplot.hist(X)
pyplot.show()
pyplot.close()
####################################################################
####  Models to predict data  ###############

# Inicialize linear models
alphas_ridge = np.logspace(-3, 3)
alphas_lasso = np.logspace(-10, 3)
lnlg = LinearRegression()
k = 10
kf = KFold(n_splits=k)
kf.get_n_splits(X)

# Using sklearn K-fold Cross validation algorithm for linear regression model
print("LINEAR REGRESSION MODEL:\n")
SSE_linalg = []
Folds = []
i = 1
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    Ytrain, Ytest = Y[train_index], Y[test_index]
    lnlg.fit(Xtrain, Ytrain)
    pred_values = lnlg.predict(Xtest)
    # ssse = lnlg.score(Xtrain, Ytrain)
    sse_linalg = np.sum(np.subtract(Ytest, pred_values) ** 2)
    SSE_linalg.append(sse_linalg)
    Folds.append("Fold " + str(i))
    print("Fold: %2d, Sum of Squared Errors with Sklearn: %.3f " % (i, sse_linalg))
    i += 1
print("Mean of SSE: ", np.mean(SSE_linalg))
print("Standard deviation of SSE: ", np.std(SSE_linalg))
plot_linear()

# Using sklearn K-fold Cross validation algorithm for Ridge regression model
print("\nRIDGE REGRESSION MODEL:\n")
SSE_ridge = []
Alpha_ridge = []
i = 1
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    Ytrain, Ytest = Y[train_index], Y[test_index]
    ridgeAlpha = RidgeCV(alphas=alphas_ridge).fit(Xtrain, Ytrain)
    ridge = Ridge(alpha=ridgeAlpha.alpha_, random_state=0)
    ridge.fit(Xtrain, Ytrain)
    pred_values = ridge.predict(Xtest)
    sse_ridge = np.sum(np.subtract(Ytest, pred_values) ** 2)
    SSE_ridge.append(sse_ridge)
    Alpha_ridge.append(ridgeAlpha.alpha_)
    print("Fold: %2d, Sum of Squared Errors with Sklearn: %.3f " % (i, sse_ridge))
    i += 1
print("Mean of SSE: ", np.mean(SSE_ridge))
print("Standard deviation of SSE: ", np.std(SSE_ridge))
print("Alphas: ", Alpha_ridge)
print("Mean of Alphas: ", np.mean(Alpha_ridge))
print("Standard deviation of Alphas: ", np.std(Alpha_ridge))
plot_ridge_sse()
plot_ridge_alpha()

# Using sklearn K-fold Cross validation algorithm for best parameter of Ridge Regression
print("\nRIDGE REGRESSION MODEL FINAL VALUES:\n")
SSE_ridge = []
alpha = np.mean(Alpha_ridge)
ridge = Ridge(alpha=alpha, random_state=0)
i = 1
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    Ytrain, Ytest = Y[train_index], Y[test_index]
    ridge.fit(Xtrain, Ytrain)
    pred_values = ridge.predict(Xtest)
    sse_ridge = np.sum(np.subtract(Ytest, pred_values) ** 2)
    SSE_ridge.append(sse_ridge)
    print("Fold: %2d, Sum of Squared Errors with Sklearn: %.3f " % (i, sse_ridge))
    i += 1
print("Mean of SSE: ", np.mean(SSE_ridge))
print("Standard deviation of SSE: ", np.std(SSE_ridge))
plot_ridge_sse()

# Using sklearn K-fold Cross validation algorithm for linear regression model
print("\nLASSO REGRESSION MODEL:\n")
SSE_lasso = []
Alpha_lasso = []
i = 1
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    Ytrain, Ytest = Y[train_index], Y[test_index]
    lassoAlpha = LassoCV(alphas=alphas_lasso).fit(Xtrain, column_or_1d(Ytrain, warn=False))
    lasso = Lasso(alpha=lassoAlpha.alpha_, random_state=0)
    lasso.fit(Xtrain, Ytrain)
    pred_values = lasso.predict(Xtest)
    sse_lasso = np.sum(np.subtract(Ytest, pred_values)**2)
    SSE_lasso.append(sse_lasso)
    Alpha_lasso.append(lassoAlpha.alpha_)
    print("Fold: %2d, Sum of Squared Errors with Sklearn: %.3f " % (i, sse_lasso))
    i += 1
print("Mean of SSE: ", np.mean(SSE_lasso))
print("Standard deviation of SSE: ", np.std(SSE_lasso))
print("Alphas: ", Alpha_lasso)
print("Mean of Alphas: ", np.mean(Alpha_lasso))
print("Standard deviation of Alphas: ", np.std(Alpha_lasso))

# Using sklearn K-fold Cross validation algorithm for linear regression model
print("\nLASSO REGRESSION FINAL VALUES:\n")
SSE_lasso = []
alphalasso = np.mean(Alpha_lasso)
lasso = Lasso(alpha=alphalasso, random_state=0)
i = 1
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    Ytrain, Ytest = Y[train_index], Y[test_index]
    lasso.fit(Xtrain, Ytrain)
    pred_values = lasso.predict(Xtest)
    sse_lasso = np.sum(np.subtract(Ytest, pred_values)**2)
    SSE_lasso.append(sse_lasso)
    Alpha_lasso.append(lassoAlpha.alpha_)
    print("Fold: %2d, Sum of Squared Errors with Sklearn: %.3f " % (i, sse_lasso))
    i += 1
print("Mean of SSE: ", np.mean(SSE_lasso))
print("Standard deviation of SSE: ", np.std(SSE_lasso))

if np.mean(SSE_linalg) <= np.mean(SSE_ridge) and np.mean(SSE_linalg) <= np.mean(SSE_lasso):
    print("LINALG")
    # test to all the data and create the final predictor
    Ytest_lin = LinearRegression().fit(X, Y).predict(X_test)
elif np.mean(SSE_lasso) <= np.mean(SSE_linalg) and np.mean(SSE_lasso) <= np.mean(SSE_ridge):
    print("LASSO")
    # test to all the data and create the final predictor
    Ytest_lin = Lasso(alpha=alphalasso, random_state=0).fit(X, Y).predict(X_test)
elif np.mean(SSE_ridge) <= np.mean(SSE_linalg) and np.mean(SSE_ridge) <= np.mean(SSE_lasso):
    print("RIDGE")
    # test to all the data and create the final predictor
    Ytest_lin = Ridge(alpha=alpha, random_state=0).fit(X, Y).predict(X_test)

# Create output file
np.save(output_dir + "/Ytest_Regression1", Ytest_lin)
