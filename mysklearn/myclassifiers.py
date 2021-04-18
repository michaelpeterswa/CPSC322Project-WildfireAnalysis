import mysklearn.myutils as myutils
import copy

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        # convert the X_train to a simple list
        x_list = [] 
        for x in X_train:
            x_list.append(x[0])
        m, b = myutils.compute_slope_intercept(x_list, y_train)
        # correlation_coeficient = myutils.correlation_coeficient(x_list, y_train)
        self.slope = m
        self.intercept = b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        x_list = []
        y_predicted = []
        for x in X_test:
            x_list.append(x[0])
        if self.slope != None and self.intercept != None:
            for x in x_list:
                y_predicted.append(x * self.slope + self.intercept)
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for test_instance in X_test:
            instance_distances = []
            for x in self.X_train:
                instance_distances.append(myutils.compute_distance_between_lists(x, test_instance))
            instance_distances, instance_indices = myutils.get_k_smallest_elements(instance_distances, 
                self.n_neighbors)
            distances.append(instance_distances)
            neighbor_indices.append(instance_indices)
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        distances, neighbor_indices = self.kneighbors(X_test)
        for row in neighbor_indices:
            frequency = {}
            for i in row:
                if self.y_train[i] in frequency:
                    frequency[self.y_train[i]] += 1
                else:
                    frequency[self.y_train[i]] = 1
            max_val = -1
            max_key = ""
            for key in frequency:
                if frequency[key] > max_val:
                    max_val = frequency[key]
                    max_key = key
            predictions.append(max_key)
        
        return predictions

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train

        # get count of priors
        self.priors = {}
        for value in y_train:
            if value in self.priors:
                self.priors[value] += 1
            else:
                self.priors[value] = 1

        # get posteriors
        # create a template dictionary
        template = []
        for row in X_train:
            for i,x in enumerate(row):
                if i >= len(template):
                    template.append({})
                if x not in template[i]:
                    template[i][x] = 0
        self.posteriors = {}
        for x in self.priors:
            self.posteriors[x] = myutils.deep_copy_item(template)
        
        # fill out the template
        for row, y in zip(X_train, y_train):
            for i,x in enumerate(row):
                self.posteriors[y][i][x] += 1

        # balance out the posteriors
        for key in self.posteriors:
            for i, row in enumerate(self.posteriors[key]):
                for key2 in row:
                    self.posteriors[key][i][key2] /= self.priors[key] 
        
        # balance out the priors
        for key in self.priors:
            self.priors[key] /= len(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        results = []
        for test in X_test:

            # get each probability
            probabilities = {}
            for key in self.posteriors:
                probabilities[key] = self.priors[key]
                for i,x in enumerate(test):
                    try:
                        probabilities[key] *= self.posteriors[key][i][x]
                    except:
                        pass
            
            # get max value
            max_key = ""
            max_value = -1
            for key in probabilities:
                if probabilities[key] > max_value:
                    max_key = key
                    max_value = probabilities[key]
            results.append(max_key)     

        return results
        
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(y_train)

        X_train2 = copy.deepcopy(X_train)
        # construct a dictionary og possible values in the form {attribute: values}
        available_attributes = {}
        for i in range(0, len(X_train[0])):
            att = "att"+str(i)
            available_attributes[att] = []
            for x in X_train:
                if x[i] not in available_attributes[att]:
                    available_attributes[att].append(x[i])

        for i,x in enumerate(y_train):
            X_train2[i].append(x)
        tree = myutils.tdidt(X_train2, [x for x in range(0, len(X_train2[0])-1)], available_attributes)
        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = []
        predictions = []
        for i in range(0, len(X_test[0])):
            header.append("att" + str(i))
        for instance in X_test:
            prediction = myutils.tdidt_predict(header, self.tree, instance)
            predictions.append(prediction)
        return predictions


    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        default_header = ["att"+str(i) for i in range(0, len(self.X_train))]
        myutils.tdidt_print_rules(self.tree, "", class_name, default_header, attribute_names)

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
