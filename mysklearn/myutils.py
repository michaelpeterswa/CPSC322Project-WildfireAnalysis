import math
import copy
import random

# gets the column from the table
def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []

    for row in table:
        if (row[col_index] != "NA"):
            col.append(row[col_index])
    return col 

# gets the frequencies from the column
def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)

    values = []
    counts = []

    for value in col:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        else:
            i = values.index(value)
            counts[i] += 1

    return values, counts 

# gets the sum of the column
def get_sum_of_column(table, header, col_name):
    col = get_column(table, header, col_name)
    return sum(col)

# gorups the column into sections
def group_by(table, header, group_by_col_name):
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)

    group_names = sorted(list(set(col)))
    group_subtables = [[] for _ in group_names]

    for row in table:
        group_by_value = row[col_index]
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row)
    return group_names, group_subtables

# computes the equal width cutoffs for a list of values
def compute_equal_width_cutoffs(values, num_bins):
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins
    
    cutoffs = []
    cutoffs.append(min(values))
    for i in range(0, num_bins-1):
        cutoffs.append(cutoffs[i] + bin_width)
    cutoffs.append(max(values))
    
    return cutoffs

# returns the frequencies from the list of values and cutoffs
def get_frequencies_from_cutoffs(values, cutoffs):
    values.sort()
    frequencies = []
    cutoff_index = 0
    value_index = 0

    while cutoff_index < len(cutoffs) - 1:
        while value_index < len(values) and values[value_index] <= cutoffs[cutoff_index+1]:
            try:
                value_index += 1
                frequencies[cutoff_index] += 1
                t = (values[value_index], cutoff_index + 1)
            except:
                frequencies.append(1)
        cutoff_index += 1
    return frequencies

# computes the slope intercept
def compute_slope_intercept(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    b = mean_y - m * mean_x
    return m, b 

# computes the correllation coeficient
def correlation_coeficient(x, y):
    mean_x = sum(x)/len(x)
    mean_y = sum(y)/len(y)
    n = len(x)
    
    numerator = sum([i*j for i, j in zip(x,y)]) - n * mean_x * mean_y
    denominator = math.sqrt((sum([i**2 for i in x]) - n * mean_x ** 2)*(sum([i ** 2 for i in y]) - n * mean_y ** 2))
    
    coeficient = numerator / denominator
    
    return coeficient

# computes the covariance
def covariance(x, y):
    mean_x = sum(x)/len(x)
    mean_y = sum(y)/len(y)
    n = len(x)
    
    numerator = sum([i*j for i, j in zip(x,y)]) - n * mean_x * mean_y
    denominator = n
    
    covariance = numerator / denominator
    
    return round(covariance, 2)

# takes in two arrays of equal length and removes elemenst where one is a missing value
def remove_missing_values_from_parral_arrays(array1, array2):
    to_delete = []
    for i, x in enumerate(array1):
        if x == "" or array2[i] == "":
            to_delete.append(i)
    to_delete.reverse()
    for x in to_delete:
        del array1[x]
        del array2[x]

# splits the data into genres
def split_into_genres(data_col, genre_col, genres):
    ratings_by_genre = {}
    for g in genres:
        ratings_by_genre[g] = []
    for i, x in enumerate(data_col):
        if x != "":
            for g in genres:
                if g in genre_col[i]:
                    ratings_by_genre[g].append(x)
    data = []
    for g in genres:
        data.append(ratings_by_genre[g])
    return data

# computes the distance between two lists
def compute_distance_between_lists(list_1, list_2):

    is_num = True
    try:
        float(list_1[0])
    except:
        is_num = False

    totals = []
    for x,y in zip(list_1, list_2):
        if is_num:
            totals.append((x-y)*(x-y))
        else:
            if x != y:
                totals.append(1)

    return math.sqrt(sum(totals))

# gets the k smallest elements
def get_k_smallest_elements(arr, k):
    arr_copy = copy.deepcopy(arr)
    arr_copy = [[x, i] for i, x in enumerate(arr_copy)]
    smallest_k = []
    smallest_indices = []

    # sort by distance, get the top k, then sort by index
    arr_copy.sort(key = lambda x: x[0])
    arr_copy = arr_copy[:k]
    arr_copy.sort(key = lambda x: x[1])
            
    for i in range(k):
        smallest_indices.append(arr_copy[i][1])
        smallest_k.append(arr_copy[i][0])

    return smallest_k, smallest_indices

# gets the DOE class from the mpg value
def get_DOE_mpg_class(mpg):
    if mpg <= 13:
        return 1
    elif mpg <= 14:
        return 2
    elif mpg <= 16:
        return 3
    elif mpg <= 19:
        return 4
    elif mpg <= 23:
        return 5
    elif mpg <= 26:
        return 6
    elif mpg <= 30:
        return 7
    elif mpg <= 36:
        return 8
    elif mpg <= 44:
        return 9
    else:
        return 10

# normalizes the columns
def normalize_column(col):
    normalized_column = []
    for x in col:
        normalized_column.append((x - min(col)) / ((max(col) - min(col)) * 1.0))
    return normalized_column

# gets the X_train, X_test, y_train, y_test from the folds
def get_trains_and_tests(X, y, X_train_fold, X_test_fold):
    X_train = [X[x] for x in X_train_fold]
    y_train = [y[x] for x in X_train_fold]
    X_test = [X[x] for x in X_test_fold]
    y_test = [y[x] for x in X_test_fold]
    return X_train, X_test, y_train, y_test

# expands the confusion matrix
def build_confusion_matrix(mat):
    for i in range(0, len(mat)):
        recognition = 0
        total = 0
        for j in range(0, len(mat[i])):
            if i == j:
                recognition += mat[i][j]
            total += mat[i][j]
        if total != 0:
            recognition = round((recognition / total) * 100, 2)
        mat[i].insert(0, i+1)
        mat[i].append(total)
        mat[i].append(recognition)

# gets the accuracy from the predicted and actual rows
def get_accuracy_from_predicted_and_actual(predicted, actual):
    acc = [int(x==y) for x,y in zip(predicted, actual)]
    acc = round(sum(acc)/len(acc), 2)
    return acc

def deep_copy_item(item):
    return copy.deepcopy(item)

# converts weight to the NHTSA rankings
def NHTSA_rankings(array):
    new_array = []
    for x in array:
        if x >= 3500:
            new_array.append(5)
        elif x >= 3000:
            new_array.append(4)
        elif x >= 2500:
            new_array.append(3)
        elif x >= 2000:
            new_array.append(2)
        else:
            new_array.append(1)
    return new_array

def all_same_class(instances):
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label

def select_attribute(instances, att_indexes, available_attributes):
    attribute_array = []

    # for each avalailable index
    for i in att_indexes:
        attributes = {}

        # get the frequency of each value
        # eg: {senior: {true: 1, false, 2}, mid: {true: 3, false, 3}}
        for value in available_attributes["att"+str(i)]:
            attributes[value] = {}
        for instance in instances:
            att_val = instance[i]
            if instance[-1] in attributes[att_val]:
                attributes[att_val][instance[-1]] += 1
            else:
                attributes[att_val][instance[-1]] = 1

        attribute_array.append(attributes)

    # loop through all the attributes and get the smallest weighted sum of entropy
    smallest_not_set = True
    smallest = 0
    smallest_index = 0
    for i, attributes in enumerate(attribute_array):
        
        weighted_sum = 0

        # get the entropy of each
        for key in attributes:
            vals = []
            for key2 in attributes[key]:
                vals.append(attributes[key][key2])
            
            entropy_sum = 0
            for v in vals:
                entropy = -((v / sum(vals)) * math.log((v/ sum(vals)), 2))
                entropy_sum += entropy 
            weighted_sum += entropy_sum * (sum(vals) / len(instances))

        if weighted_sum <= smallest or smallest_not_set:
            smallest = weighted_sum
            smallest_index = i
            smallest_not_set = False
        
    return att_indexes[smallest_index]

def compute_partition_stats(instances, class_index):
    stats = {}
    for x in instances:
        if x[class_index] in stats:
            stats[x[class_index]] += 1
        else:
            stats[x[class_index]] = 1
    stats_array = []
    # print(stats)
    for key in stats:
        stats_array.append([key, stats[key], len(instances)])
    
    return stats_array
    

def tdidt(current_instances, att_indexes, att_domains):

    # print(att_indexes)
    split_attribute = select_attribute(current_instances, att_indexes, att_domains)
    # print("TEST", split_attribute, "T", att_indexes)
    class_label = "att"+str(split_attribute)
    att_indexes2 = copy.deepcopy(att_indexes)
    att_indexes2.remove(split_attribute)
    
    partitions = {}
    attributes = att_domains[class_label]
    for a in attributes:
        partitions[a] = []
    for instance in current_instances:
        partitions[instance[split_attribute]].append(instance)
    
    tree = ["Attribute", "att"+str(split_attribute)]

    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]

        if len(partition) > 0 and all_same_class(partition):
            leaf = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(att_indexes2) == 0:
            partition_stats = compute_partition_stats(partition, -1)
            partition_stats.sort(key=lambda x: x[1])
            leaf = ["Leaf", partition_stats[-1][0], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
            
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            partition_stats = compute_partition_stats(current_instances, -1)
            partition_stats.sort(key=lambda x: x[1])
            leaf = ["Leaf", partition_stats[-1][0], len(partition), len(current_instances)]
            return leaf
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, att_indexes2, att_domains)
            values_subtree.append(subtree)
            tree.append(values_subtree)
    return tree

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        # now I need to find which "edge" to follow recursively
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match!! recurse!!
                return tdidt_predict(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label

def tdidt_print_rules(tree, rule, class_name, default_header, attribute_names):
    info_type = tree[0]
    if info_type == "Attribute":
        if rule != "":
            rule += " AND "
        else:
            rule += "IF "
        if attribute_names is None: 
            rule += tree[1]
        else:
            index = default_header.index(tree[1])
            rule += attribute_names[index]
            
        for i in range(2, len(tree)):
            value_list = tree[i]
            rule2 = rule + " = " + str(value_list[1])
            tdidt_print_rules(value_list[2], rule2, class_name, default_header, attribute_names)
    else: # "Leaf"
        print(rule, "THEN", class_name, "=", tree[1])