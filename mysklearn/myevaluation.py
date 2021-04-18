import mysklearn.myutils as myutils
import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    if random_state is not None:
        np.random.seed(random_state)
    
    if shuffle: 
        for i in range(len(X)):
            rand_index = np.random.randint(0, len(X)) # [0, len(alist))
            X[i], X[rand_index] = X[rand_index], X[i]
            y[i], y[rand_index] = y[rand_index], y[i]

    split_index = 0
    if test_size < 1:
        split_index = int(len(X) * test_size) + 1
    else:
        split_index = test_size
    split_index = len(X) - split_index
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]


    
    return X_train, X_test, y_train, y_test # TODO: fix this

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    indices = [x for x in range(0, len(X))]

    if random_state is not None:
        np.random.seed(random_state)
    
    if shuffle: 
        for i in range(len(X)):
            rand_index = np.random.randint(0, len(X)) # [0, len(alist))
            indices[i], indices[rand_index] = indices[rand_index], indices[i]

    X_test_folds = [[] for _ in range(0, n_splits)]
    samples_mod_splits = len(X) % n_splits
    index = 0
    for i in indices:
        if index < samples_mod_splits:
            if len(X_test_folds[index]) > (len(X) // n_splits):
                index += 1
        else:
            if len(X_test_folds[index]) > (len(X) // n_splits) - 1:
                index += 1
        X_test_folds[index].append(i)
            
    X_train_folds = [[] for _ in range(0, n_splits)]
    for i in range(0, len(X)):
        for j in range(0, n_splits):
            if i not in X_test_folds[j]:
                X_train_folds[j].append(i)
    
    return X_train_folds, X_test_folds # TODO: fix this

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """

    indices = [x for x in range(0, len(X))]

    if random_state is not None:
        np.random.seed(random_state)
    
    if shuffle: 
        for i in range(len(X)):
            rand_index = np.random.randint(0, len(X)) # [0, len(alist))
            indices[i], indices[rand_index] = indices[rand_index], indices[i]

    classifications = []
    unigue_classifiers = []
    for i,c in enumerate(y):
        if c in unigue_classifiers:
            classifications[unigue_classifiers.index(c)].append(indices[i])
        else:
            classifications.append([indices[i]])
            unigue_classifiers.append(c)
    
    index = 0
    X_test_folds = [[] for _ in range(0, n_splits)]
    for c in classifications:
        for d in c:
            X_test_folds[index%n_splits].append(d)
            index += 1
    
    X_train_folds = [[] for _ in range(0, n_splits)]
    for i in range(0, len(X)):
        for j in range(0, n_splits):
            if i not in X_test_folds[j]:
                X_train_folds[j].append(i)

    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0 for _ in labels] for _ in labels]
    for x, y in zip(y_true, y_pred):
        matrix[labels.index(x)][labels.index(y)] += 1
    return matrix