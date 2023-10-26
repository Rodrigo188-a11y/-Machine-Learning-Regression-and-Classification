import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import plot_confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

IMG_SIZE = (5, 5, 3)
PIXEL_NO = 75


def see_images_from_array(x, y):  # plots the images

    for i in range(len(y)):  # searches for background, center and ring
        if y[i] == 2:
            image = i
            break

    img = np.reshape(x[image], IMG_SIZE)  # reshapes the array to the correct form

    image2 = Image.fromarray(img)  # create Pillow image from array
    pyplot.imshow(image2)
    pyplot.show()
    pyplot.close()


def tweakImage(img):  # randomly rotates an image, and randomly reflects it
    random.seed(42)
    new_image = np.reshape(img, IMG_SIZE)

    r1 = random.randint(0, 3)
    r2 = random.random()
    r3 = random.randint(0, 1)

    # Rotate random number of times
    for i in range(r1):
        new_image = np.rot90(new_image)

    # Mirror randomly
    if r2 > 0.49:
        np.flip(new_image, r3)

    new_image = np.reshape(new_image, (1, PIXEL_NO))

    return new_image


def underSample_all_to_min(x, y):  # responsible for undersampling the two biggest sets to the size of the smallest
    indices_background = []
    indices_ring = []
    indices_center = []
    background_count = 0
    ring_count = 0
    center_count = 0

    for i in range(len(y)):  # counts the number of appearences of each set and their indexes
        if y[i] == 0:
            indices_background.append(i)
            background_count += 1
        elif y[i] == 1:
            indices_ring.append(i)
            ring_count += 1
        else:
            indices_center.append(i)
            center_count += 1

    if background_count <= ring_count and background_count <= center_count:  # sees if background is the lowest set

        # undersamples both other two sets to background size
        diff = ring_count - background_count
        new_x, new_y, indices_background, indices_ring, indices_center = undersample(indices_ring, diff, x, y)

        diff = center_count - background_count
        new_x, new_y, indices_background, indices_ring, indices_center = undersample(indices_center, diff, new_x, new_y)
    elif ring_count <= background_count and ring_count <= center_count:  # sees if ring is the lowest set
        diff = background_count - ring_count
        new_x, new_y, indices_background, indices_ring, indices_center = undersample(indices_background, diff, x, y)

        diff = center_count - ring_count
        new_x, new_y, indices_background, indices_ring, indices_center = undersample(indices_center, diff, new_x, new_y)
    else:  # sees if center is the lowest set
        diff = background_count - center_count
        new_x, new_y, indices_background, indices_ring, indices_center = undersample(indices_background, diff, x, y)

        diff = ring_count - center_count
        new_x, new_y, indices_background, indices_ring, indices_center = undersample(indices_ring, diff, new_x, new_y)

    return [new_x, new_y]


def overSample_underSample_to_mid(x, y):  # responsible for undersampling the biggest set and oversampling the lowest to
    # the size of the medium

    indices_background = []
    indices_ring = []
    indices_center = []
    background_count = 0
    ring_count = 0
    center_count = 0

    for i in range(len(y)):  # counts the number of appearences of each set and their indexes
        if y[i] == 0:
            indices_background.append(i)
            background_count += 1
        elif y[i] == 1:
            indices_ring.append(i)
            ring_count += 1
        else:
            indices_center.append(i)
            center_count += 1

    if background_count >= ring_count and background_count >= center_count:   # sees if background is the biggest set
        if ring_count > center_count:  # sees if ring is the second biggest set
            # undersamples background
            subtract = background_count - ring_count
            x, y, indices_background, indices_ring, indices_center = undersample(indices_background, subtract, x, y)
            # oversamples center
            diff = ring_count - center_count
            random_indices = np.random.choice(indices_center, diff)
        else:  # sees if center is the second biggest set
            subtract = background_count - center_count
            x, y, indices_background, indices_ring, indices_center = undersample(indices_background, subtract, x, y)
            diff = center_count - ring_count
            random_indices = np.random.choice(indices_ring, diff)

        new_x = x
        new_y = y
        new_x, new_y = overSample(diff, x, y, new_x, new_y, random_indices)

    elif ring_count >= background_count and ring_count >= center_count:  # sees if ring is the biggest set
        if background_count > center_count:
            subtract = ring_count - background_count
            x, y, indices_background, indices_ring, indices_center = undersample(indices_ring, subtract, x, y)
            diff = background_count - center_count
            random_indices = np.random.choice(indices_center, diff)
        else:
            subtract = ring_count - center_count
            x, y, indices_background, indices_ring, indices_center = undersample(indices_ring, subtract, x, y)
            diff = center_count - background_count
            random_indices = np.random.choice(indices_background, diff)
        new_x = x
        new_y = y
        new_x, new_y = overSample(diff, x, y, new_x, new_y, random_indices)
    else:  # sees if center is the biggest set
        if background_count > ring_count:
            subtract = center_count - background_count
            x, y, indices_background, indices_ring, indices_center = undersample(indices_center, subtract, x, y)
            diff = background_count - ring_count
            random_indices = np.random.choice(indices_ring, diff)
        else:
            subtract = center_count - ring_count
            x, y, indices_background, indices_ring, indices_center = undersample(indices_center, subtract, x, y)
            diff = ring_count - background_count
            random_indices = np.random.choice(indices_background, diff)
        new_x = x
        new_y = y
        new_x, new_y = overSample(diff, x, y, new_x, new_y, random_indices)

    return [new_x, new_y]


def overSample_all_to_max(x, y):  # responsible for oversampling the two smallest sets to the size of the biggest
    indices_background = []
    indices_ring = []
    indices_center = []
    background_count = 0
    ring_count = 0
    center_count = 0

    new_x = x
    new_y = y

    for i in range(len(y)):  # counts the number of appearences of each set and their indexes
        if y[i] == 0:
            indices_background.append(i)
            background_count += 1
        elif y[i] == 1:
            indices_ring.append(i)
            ring_count += 1
        else:
            indices_center.append(i)
            center_count += 1

    if background_count >= ring_count and background_count >= center_count:  # sees if background is the biggest set

        # oversamples both the others to the biggest set size
        diff = ring_count - background_count
        new_x, new_y = overSample(diff, x, y, new_x, new_y, indices_background)

        diff = ring_count - center_count
        new_x, new_y = overSample(diff, x, y, new_x, new_y, indices_center)

    elif ring_count >= background_count and ring_count >= center_count:  # sees if ring is the biggest set

        diff = ring_count - background_count
        new_x, new_y = overSample(diff, x, y, new_x, new_y, indices_background)

        diff = ring_count - center_count
        new_x, new_y = overSample(diff, x, y, new_x, new_y, indices_center)
    else:  # sees if center is the biggest set

        diff = center_count - background_count
        new_x, new_y = overSample(diff, x, y, new_x, new_y, indices_background)

        diff = center_count - ring_count
        new_x, new_y = overSample(diff, x, y, new_x, new_y, indices_ring)

    return [new_x, new_y]


def undersample(indices, n_samples_to_remove, x, y):  # Delete samples from the sets that need to be decreased
    # used to randomize the indexes from the set to be decreased
    random_indices = random.sample(indices, n_samples_to_remove)

    # deletes the indexes from the data
    x = np.delete(x, random_indices, 0)
    y = np.delete(y, random_indices)

    # calculates the new indexes
    indices_background, indices_ring, indices_center = index_count(y)

    return x, y, indices_background, indices_ring, indices_center


def index_count(y):  # counts the indexes of each set and returns it
    indices_background = []
    indices_ring = []
    indices_center = []
    background_count = 0
    ring_count = 0
    center_count = 0

    for i in range(len(y)):
        if y[i] == 0:
            indices_background.append(i)
            background_count += 1
        elif y[i] == 1:
            indices_ring.append(i)
            ring_count += 1
        else:
            indices_center.append(i)
            center_count += 1
    return indices_background, indices_ring, indices_center


def overSample(n_samples_to_add, x, y, new_x, new_y, indices):  # Create samples to the sets that need to be bigger
    # used to randomize the indexes from the set to be bigger
    random_indices = np.random.choice(indices, n_samples_to_add)

    for i in range(n_samples_to_add):  # creates new images from that set by reflecting and rotating current ones
        new_image = x[random_indices[i]]
        new_image = tweakImage(new_image)

        new_x = np.append(new_x, new_image, axis=0)
        new_y = np.append(new_y, y[random_indices[i]])
    return new_x, new_y


def nusvc(x_train, x_test, y_train, y_test, fold_number, model_key, hyper):
    global models, clf_last

    if hyper:  # tests multi parameters to find the best ones to use
        nu = [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.18, 0.2, 0.19, 0.22, 0.21, 0.23, 0.24]
        parameters = {'nu': nu, 'kernel': ['rbf', 'poly', 'sigmoid'], 'random_state': [4],
                      'degree': [2, 3, 4], 'class_weight': ['balanced'], 'decision_function_shape': ['ovo']}

        clf = GridSearchCV(NuSVC(), parameters, n_jobs=-1, scoring='balanced_accuracy', cv=5)

        clf.fit(x_train, y_train)

        # checks if the hyperparameter where already found in previous runs and if it does, won't introduce this model
        # again for final testing
        error = 0
        for i in range(len(clf_last)):
            if clf.best_params_ == clf_last[i]:
                error = 1
                print("Model " + 'nusvc_fold' + str(fold_number + 1) + " already exists.")
                break
        # if it doens't exist, the model is introduced in the final models to be evaluated
        if error == 0:
            clf_last.append(clf.best_params_)
            classifier = 'nusvc_fold' + str(fold_number + 1)
            print("Model " + classifier, clf.score(x_train, y_train))
            models_aux1 = {'best params': clf.best_params_, 'score': 0}
            models[classifier] = models_aux1
    else:  # final model evaluation. It will compute the balanced_accuracy over n number of runs to see average accuracy
        print("NUSVC")
        best_params = models[model_key]["best params"]

        clf = NuSVC(**best_params)

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        scores(clf, y_pred, x_test, y_test, model_key)


def kneighbors(x_train, x_test, y_train, y_test, fold_number, model_key, hyper):
    global models, clf_last

    if hyper:  # tests multi parameters to find the best ones to use
        leaf_size = np.arange(20, 51)
        n_neighbors = np.arange(2, 11)
        parameters = {'n_neighbors': n_neighbors, 'weights': ['distance'],
                      'leaf_size': leaf_size, 'p': [1, 2]}

        clf = GridSearchCV(KNeighborsClassifier(), parameters, n_jobs=-1, scoring='balanced_accuracy', cv=5)

        clf.fit(x_train, y_train)

        # checks if the hyperparameter where already found in previous runs and if it does, won't introduce this model
        # again for final testing
        error = 0
        for i in range(len(clf_last)):
            if clf.best_params_ == clf_last[i]:
                error = 1
                print("Model " + 'kneighbors_fold' + str(fold_number + 1) + " already exists.")
                break
        # if it doens't exist, the model is introduced in the final models to be evaluated
        if error == 0:
            clf_last.append(clf.best_params_)
            classifier = 'kneighbors_fold' + str(fold_number + 1)
            print("Model " + classifier, clf.score(x_train, y_train))
            models_aux1 = {'best params': clf.best_params_, 'score': 0}
            models[classifier] = models_aux1

    else:  # final model evaluation. It will compute the balanced_accuracy over n number of runs to see average accuracy
        print("KNEI")
        best_params = models[model_key]["best params"]

        clf = KNeighborsClassifier(**best_params)

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        scores(clf, y_pred, x_test, y_test, model_key)


def decisiontrees(x_train, x_test, y_train, y_test, fold_number, model_key, hyper):
    global models, clf_last

    if hyper: # tests multi parameters to find the best ones to use
        parameters = {'criterion': ['gini', 'entropy', 'log_loss'], 'class_weight': ['balanced'], 'random_state': [4],
                      'splitter': ['best', 'random'], 'max_features': [None, 'sqrt', 'log2']}

        clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=-1, scoring='balanced_accuracy', cv=5)

        clf.fit(x_train, y_train)

        # checks if the hyperparameter where already found in previous runs and if it does, won't introduce this model
        # again for final testing
        error = 0
        for i in range(len(clf_last)):
            if clf.best_params_ == clf_last[i]:
                error = 1
                print("Model " + 'decisiontrees_fold' + str(fold_number + 1) + " already exists.")
                break
        # if it doens't exist, the model is introduced in the final models to be evaluated
        if error == 0:
            clf_last.append(clf.best_params_)
            classifier = 'decisiontrees_fold' + str(fold_number + 1)
            print("Model " + classifier, clf.score(x_train, y_train))
            models_aux1 = {'best params': clf.best_params_, 'score': 0}
            models[classifier] = models_aux1

    else:  # final model evaluation. It will compute the balanced_accuracy over n number of runs to see average accuracy
        print("TREE")
        best_params = models[model_key]["best params"]
        clf = DecisionTreeClassifier(**best_params)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        scores(clf, y_pred, x_test, y_test, model_key)


def scores(clf, y_pred, x_test, y_test, model_key):  # plots the confusion matrix and the balance accuracy score and
    # stores in the model
    global models

    fig = plot_confusion_matrix(clf, x_test, y_test, display_labels=clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for Binary Classification")
    plt.show()
    plt.close()
    print("BINARY PROBLEM SCORES NN: ")
    print(classification_report(y_test, y_pred))
    score_aux = balanced_accuracy_score(y_pred=y_pred, y_true=y_test)
    print(score_aux)
    if not model_key is None:
        models[model_key]["score"] = score_aux + models[model_key]["score"]
        models[model_key]["model"] = clf


# get current directory
cur_dir = os.getcwd()
cur_dir1 = cur_dir + '/data'

X = np.load(cur_dir1 + '/Xtrain_Classification2.npy')
Y = np.load(cur_dir1 + '/Ytrain_Classification2.npy')
Xtest_geral = np.load(cur_dir1 + '/Xtest_Classification2.npy')
print("X shape: ", X.shape)
print("Xtest shape: ", Xtest_geral.shape)
print("Y shape: ", Y.shape)

# see_images_from_array(X, Y)

###########################################
#####  Classification Algorithm with  #####
#####        balanced data            #####
###########################################
models = {}  # dict that will store models' best data and is updated on the score_bin() function
clf_last = []  # stores the current existing models to prevent equal models to be introduced on hyper-param tuning phase

number_loop = 10  # number of times to undersample the data randomly and test the best hyperparameters
#### This for loop tests the best hyperparameters
# the data here is undersampled to the lowest size set, because we want to make the data smaller to test more
# hyperparameters and faster
for j in range(number_loop):
    [X_undersample, Y_undersample] = underSample_all_to_min(X, Y)  # undersamples the data
    if j == 0:
        print("---------")
        print("X shape: ", X_undersample.shape)
        print("Y shape: ", Y_undersample.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X_undersample, Y_undersample, random_state=42, train_size=.8,
                                                        stratify=Y_undersample)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Uses different models for the data
    nusvc(X_train, X_test, Y_train, Y_test, j, model_key=None, hyper=True)
    kneighbors(X_train, X_test, Y_train, Y_test, j, model_key=None, hyper=True)
    decisiontrees(X_train, X_test, Y_train, Y_test, j, model_key=None, hyper=True)
    print("\n")

print("Models with the best hyperparameters:")
for model_name, model_info in models.items():
    print("\nModel name:", model_name)
    name = model_name
    for key in model_info:
        print(key + ':', model_info[key])
        continue


number_loop2 = 5  # number of times to calculate the models' accuracy. This way we can see what models bring the best
# results over more different data
#### This for loop tests the best model
for j in range(number_loop2):
    [X_undersample_oversample, Y_undersample_oversample] = overSample_underSample_to_mid(X, Y)  # to better test the
    # models the dataset is oversampled and undersampled to medium size sets bigger, to make it more accurate but still
    # fast
    if j == 0:
        print("---------")
        print("X shape: ", X_undersample_oversample.shape)
        print("Y shape: ", Y_undersample_oversample.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X_undersample_oversample, Y_undersample_oversample,
                                                        random_state=42, train_size=.8,
                                                        stratify=Y_undersample_oversample)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for model in models:
        print(model)
        if model[:5] == "nusvc":
            nusvc(X_train, X_test, Y_train, Y_test, j, model, hyper=False)
        elif model[:10] == "kneighbors":
            kneighbors(X_train, X_test, Y_train, Y_test, j, model, hyper=False)
        else:
            decisiontrees(X_train, X_test, Y_train, Y_test, j, model, hyper=False)

# After the testing is done the model that performed the overall best is selected to predict the final data
final_name = ''
for model_name, model_info in models.items():
    print("\nModel name:", model_name)
    name = model_name
    for key in model_info:
        print(key + ':', model_info[key])
        if key == 'model':
            model = model_info[key]
        elif key == 'best params':
            params = model_info[key]
        else:
            score = model_info[key]
    if final_name == '' or score > final_score:
        final_name = name
        final_score = score
        final_model = model
        final_params = params
    else:
        continue

print("BEST MODEL: ")
print(final_name)
print(final_model)
print(final_score)
print(final_params)
# creates the biggest data set with all the data overampled to predict the Xtest_geral
try:
    X_oversample = np.load(cur_dir1 + '/Xtrain_Classification2_Oversample.npy')
    Y_oversample = np.load(cur_dir1 + '/Ytrain_Classification2_Oversample.npy')
except FileNotFoundError:
    [X_oversample, Y_oversample] = overSample_all_to_max(X, Y)
    np.save(cur_dir1 + '/Xtrain_Classification2_Oversample', X_oversample)
    np.save(cur_dir1 + '/Ytrain_Classification2_Oversample', Y_oversample)

### Used to see the final accuracy of the model on the last training dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_oversample, Y_oversample, random_state=42, train_size=.8,
                                                    stratify=Y_oversample)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("---------")
print("X shape: ", X_oversample.shape)
print("Y shape: ", Y_oversample.shape)

clf2 = final_model
clf2.fit(X_train, Y_train)
y_pred_ = clf2.predict(X_test)
print("FINAL SCORES BEST MODEL")
scores(clf2, y_pred_, X_test, Y_test, model_key=None)

fig = plot_confusion_matrix(clf2, X_test, Y_test, display_labels=clf2.classes_)
fig.figure_.suptitle("Confusion Matrix for Best Model on Oversampled Data")
plt.show()
plt.close()

### Finally predicts the intended X test for delivery and the best model is fitted to the biggest dataset to make it
# more reliable
y_predict = final_model.fit(X_oversample, Y_oversample).predict(Xtest_geral)

# Create output file
isExist = os.path.exists(cur_dir + '/output')
if not isExist:
    os.makedirs(cur_dir + '/output')

output_dir = cur_dir + '/output'
np.save(output_dir + "/Ytest_Classification2", y_predict)
