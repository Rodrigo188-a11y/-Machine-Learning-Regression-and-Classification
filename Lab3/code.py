import os
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, classification_report, f1_score


IMG_SIZE = (30, 30, 3)
PIXEL_NO = 2700


def see_images_from_array(x, y):  # plots the images
    img_size = (30, 30, 3)  # correct shape of the numpy array
    img = np.reshape(x[1001], img_size)  # reshapes the array to the correct form

    if y[1000] == 1:
        print("Eyespot")  # see if it is an eyespot(1) or a spot(0)
    else:
        print("Spot")
    image2 = Image.fromarray(img)  # create Pillow image from array
    pyplot.imshow(image2)
    pyplot.show()
    pyplot.close()


def data_indexes(y):  # stores the indexes from the Ytrain vector containing the eyespots(1) or the spots(0)
    indexes_eyespot = []  # stores eyespot indexes
    indexes_spot = []  # stores spot indexes
    eyespot_count = 0  # counts the number of eyespot ocurrences
    spot_count = 0  # counts the number of spot ocurrences

    # retrieves the indexes for each class
    for i in range(len(y)):
        if y[i] == 1:
            indexes_eyespot.append(i)
            eyespot_count += 1
        else:
            indexes_spot.append(i)
            spot_count += 1
    #print("Indexes of eyespot: ", indexes_eyespot)
    #print("Indexes of spot: ", indexes_spot)
    print("Nº. ocurrences of eyespot: ", eyespot_count)
    print("Nº. ocurrences of spot: ", spot_count)

    if eyespot_count > spot_count:
        div = eyespot_count / spot_count
        class_weight = {0: 1, 1: round(div, 2)}
    else:
        div = spot_count / eyespot_count
        class_weight = {0: round(div, 2), 1: 1}

    # creates and return a dictionary containing the data
    data_aux = {'index_eyespot': indexes_eyespot, 'index_spot': indexes_spot, 'eyespot_count': eyespot_count,
                'spot_count': spot_count, 'class_weight': class_weight}
    return data_aux


def tweakImage(img):
    random.seed(42)
    new_image = np.reshape(img, IMG_SIZE)

    r1 = random.randint(0, 3)
    r2 = random.random()
    r3 = random.randint(0, 1)

    # Rotate random number of times
    for i in range(r1):
        new_image = np.rot90(new_image)

    # Mirror randomly
    if (r2 > 0.49):
        np.flip(new_image, r3)

    new_image = np.reshape(new_image, (1, PIXEL_NO))

    return new_image


def overSample(x, y):
    indices_eyespot = []  # stores eyespot indexes
    indices_spot = []  # stores spot indexes
    eyespot_count = 0  # counts the number of eyespot ocurrences
    spot_count = 0  # counts the number of spot ocurrences

    new_x = x
    new_y = y

    for i in range(len(y)):
        if y[i] == 1:
            indices_eyespot.append(i)
            eyespot_count += 1
        else:
            indices_spot.append(i)
            spot_count += 1

    diff = eyespot_count - spot_count

    if diff == 0:
        return

    if diff > 0:
        random_indices = np.random.choice(indices_eyespot, diff)
        for i in range(diff):
            new_image = x[random_indices[i]]
            new_image = tweakImage(new_image)

            new_x = np.append(new_x, new_image, axis=0)
            new_y = np.append(new_y, y[indices_spot[i]])
    else:
        random_indices = np.random.choice(indices_eyespot, abs(diff))
        for i in range(abs(diff)):
            new_image = x[random_indices[i]]
            new_image = tweakImage(new_image)

            new_x = np.append(new_x, new_image, axis=0)
            new_y = np.append(new_y, y[indices_eyespot[i]])

    return [new_x, new_y]


def showImage(img):
    image2 = Image.fromarray(img)  # create Pillow image from array
    pyplot.imshow(image2)
    pyplot.show()
    pyplot.close()


def binary_problem_nn_unbalanced(x_train, x_test, y_train, y_test):
    # Testa hyperparametros para o training set normal, o melhor foi 'activation': 'relu', 'hidden_layer_sizes':
    # (6, 3, 2), 'learning_rate': 'constant', 'max_iter': 2000, 'random_state': 4, 'solver': 'sgd'
    """
    parameters = {'hidden_layer_sizes': [(100, 50, 20)],
                  'max_iter': [2000],
                  'random_state': [4],
                  'activation': ['relu'],
                  'solver': ['adam'],
                  'learning_rate': ['constant']}

    clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, scoring='f1_macro', cv=5)
    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.best_params_)
    """

    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(100, 50, 20), learning_rate='constant', max_iter=2000,
                        random_state=4, solver='adam')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(x_test)

    scores_bin(clf, y_pred, x_test, y_test, 'nn')


def binary_problem_svc_unbalanced(x_train, x_test, y_train, y_test):
    # Testa hyperparametros para o training set normal, o melhor foi

    """
    C = np.arange(20, 120, 20)

    parameters = {'C': C, 'kernel': ['rbf','sigmoid'], 'random_state': [4],
                  'degree': [3], 'class_weight': ['balanced']}

    # lbfgs doesn't converge; best results with just one hidden layer
    clf = GridSearchCV(SVC(), parameters, n_jobs=-1, scoring='f1_macro', cv=5)

    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.best_params_)
    """

    clf = SVC(C=10, kernel='rbf', random_state=4, degree=3, class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    scores_bin(clf, y_pred, x_test, y_test, 'svc')


def binary_problem_nusvc_unbalanced(x_train, x_test, y_train, y_test, data_information_aux):    # Testa hyperparametros para o training set normal, o melhor foi
    """
    nu = np.arange(0.1, 1, 0.3)

    parameters = {'nu': [0.2], 'kernel': ['poly', 'rbf'], 'random_state': [4],
                  'degree': [4], 'class_weight': ['balanced']}

    # lbfgs doesn't converge; best results with just one hidden layer
    clf = GridSearchCV(NuSVC(), parameters, n_jobs=-1, scoring='f1_macro', cv=5)

    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.best_params_)
    """

    clf = NuSVC(class_weight='balanced', degree=4, kernel='rbf', nu=0.2, random_state=4)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    scores_bin(clf, y_pred, x_test, y_test, 'nusvc')


def binary_problem_kneighbors_unbalanced(x_train, x_test, y_train, y_test):
    # Testa hyperparametros para o training set normal, o melhor foi
    """
    parameters = {'n_neighbors': [6], 'weights': ['distance'], 'leaf_size': [30, 50, 20]}

    # lbfgs doesn't converge; best results with just one hidden layer
    clf = GridSearchCV(KNeighborsClassifier(), parameters, n_jobs=-1, scoring='f1_macro', cv=5)

    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.best_params_)
    """

    clf = KNeighborsClassifier(n_neighbors=6, weights='distance', n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    scores_bin(clf, y_pred, x_test, y_test, 'kneighbors')


def binary_problem_decisiontrees_unbalanced(x_train, x_test, y_train, y_test):
    # Testa hyperparametros para o training set normal, o melhor foi
    """
    parameters = {'criterion': ['gini', 'entropy', 'log_loss'], 'class_weight': ['balanced'], 'random_state': [4]}

    # lbfgs doesn't converge; best results with just one hidden layer
    clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=-1, scoring='f1_macro', cv=5)

    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.best_params_)
    """

    clf = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', random_state=4)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    scores_bin(clf, y_pred, x_test, y_test, 'decisiontrees')


def binary_problem_gaussiannb_unbalanced(x_train, x_test, y_train, y_test):
    # Testa hyperparametros para o training set normal, o melhor foi

    """
    var_smoothing = np.logspace(-10, 10, num=100)
    parameters = {'var_smoothing': var_smoothing}
    print(var_smoothing)
    # lbfgs doesn't converge; best results with just one hidden layer
    clf = GridSearchCV(GaussianNB(), parameters, n_jobs=-1, scoring='f1_macro', cv=5)

    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.best_params_)
    """

    clf = GaussianNB(var_smoothing=533.6699231206302)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    scores_bin(clf, y_pred, x_test, y_test, 'gaussian')


def scores_bin(clf, y_pred, x_test, y_test, classifier):

    global models

    fig = plot_confusion_matrix(clf, x_test, y_test, display_labels=clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for Binary Classification")
    plt.show()
    plt.close()
    print("BINARY PROBLEM SCORES NN: ")
    print(classification_report(y_test, y_pred))
    score_aux = f1_score(y_test, y_pred, average='macro')
    print(score_aux)

    models_aux1 = {'model': clf, 'score': score_aux}
    models[classifier] = models_aux1


###########################################
#####  Data upload and visualization  #####
###########################################

# get current directory
cur_dir = os.getcwd()
cur_dir1 = cur_dir + '/data'

# open the files from data1 folder that contain the np arrays
X = np.load(cur_dir1 + '/Xtrain_Classification1.npy')
Y = np.load(cur_dir1 + '/ytrain_Classification1.npy')
Xtest_geral = np.load(cur_dir1 + '/Xtest_Classification1.npy')
print("X shape: ", X.shape)
print("Xtest shape: ", Xtest_geral.shape)
print("Y shape: ", Y.shape)

# helps us visualize the data
# see_images_from_array(X, Y)

# helps us see if the data is random and retieves indexes for the eyespot(1) and spots(0) values
data_information = data_indexes(Y)

###########################################
#####  Classification Algorithm with  #####
#####        balanced data            #####
###########################################

models = {}  # dict that will store models' best data and is updated on the score_bin() function

[X, Y] = overSample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, train_size=.8, stratify=Y)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#[X_train, Y_train] = overSample(X_train, Y_train)

# Use a Multi-layer Perceptron classifier to predict data
binary_problem_nn_unbalanced(X_train, X_test, Y_train, Y_test)

# Use an SVC classifier to predict data
binary_problem_svc_unbalanced(X_train, X_test, Y_train, Y_test)

# Use an NU SVC classifier to predict data
binary_problem_nusvc_unbalanced(X_train, X_test, Y_train, Y_test, data_information)

# Use an KNeighbors classifier to predict data
binary_problem_kneighbors_unbalanced(X_train, X_test, Y_train, Y_test)

# Use a Decision Tree classifier to predict data
binary_problem_decisiontrees_unbalanced(X_train, X_test, Y_train, Y_test)

# Use a Naive Gaussian classifier to predict data
binary_problem_gaussiannb_unbalanced(X_train, X_test, Y_train, Y_test)


###########################################
#####  Choose best model to finalize  #####
###########################################

final_name = ''
for model_name, model_info in models.items():
    print("\nModel name:", model_name)
    name = model_name
    for key in model_info:
        print(key + ':', model_info[key])
        if key == 'model':
            model = model_info[key]
        else:
            score = model_info[key]
    if final_name == '':
        final_name = name
        final_score = score
        final_model = model
    elif score > final_score:
        final_name = name
        final_score = score
        final_model = model
    else:
        continue

print("-------------")
print(final_name)
print(final_model)
print(final_score)
y_predict = final_model.fit(X, Y).predict(Xtest_geral)

# Create output file
isExist = os.path.exists(cur_dir + '/output')
if not isExist:
    os.makedirs(cur_dir + '/output')

output_dir = cur_dir + '/output'
np.save(output_dir + "/Ytest_Classification1", y_predict)
