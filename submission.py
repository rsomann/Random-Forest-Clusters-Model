import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        # """Get a child node based on the decision function.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︆͏︌͏󠄀
        # Args:͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︆͏︌͏󠄀
        #     feature (list(int)): vector for feature.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︆͏︌͏󠄀
        # Return:͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︆͏︌͏󠄀
        #     Class label if a leaf node, otherwise a child node.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︆͏︌͏󠄀
        # """͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︆͏︌͏󠄀
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = DecisionNode(None, None, lambda a : a[0]==1)
    decision_tree_root.right = DecisionNode(None, None, lambda a : a[3]==1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)
    
    decision_tree_root.right.right = DecisionNode(None, None, lambda a : a[2]==1)
    decision_tree_root.right.left = DecisionNode(None, None, lambda a : a[1]==1)
    
    
    decision_tree_root.right.left.right = DecisionNode(None, None, None, 1)
    decision_tree_root.right.left.left = DecisionNode(None, None, None, 0)
    
    decision_tree_root.right.right.right = DecisionNode(None, None, None, 1)
    decision_tree_root.right.right.left = DecisionNode(None, None, None, 0)
    
    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.

    Output will in the format:

                        |Predicted|
    |T|                
    |R|    [[true_positive, false_negative],
    |U|    [false_positive, true_negative]]
    |E|

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    length = len(classifier_output)
    for k in range(length):
        if classifier_output[k]==1 and true_labels[k]==1:
            true_positive+=1
        elif classifier_output[k]==1 and true_labels[k]==0:
            false_positive+=1
        elif classifier_output[k]==0 and true_labels[k]==1:
            false_negative+=1
        elif classifier_output[k]==0 and true_labels[k]==0:
            true_negative+=1


    c_matrix = [[true_positive, false_negative],
                [false_positive, true_negative]]
    
    return c_matrix


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    c_matrix = confusion_matrix(classifier_output, true_labels)
    false_pos = c_matrix[1][0]
    true_pos = c_matrix[0][0]

    precision_val = true_pos / (true_pos+false_pos)
    return precision_val


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    c_matrix = confusion_matrix(classifier_output, true_labels)
    false_neg = c_matrix[0][1]
    true_pos = c_matrix[0][0]

    recall_val = true_pos / (true_pos + false_neg)
    return recall_val


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    c_matrix = confusion_matrix(classifier_output, true_labels)
    true_pos = c_matrix[0][0]
    false_pos = c_matrix[1][0]
    false_neg = c_matrix[0][1]
    true_neg = c_matrix[1][1]

    return (true_pos+true_neg) / (true_pos+false_pos+false_neg+true_neg)


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """

    
    a = np.sum(class_vector)/len(class_vector)
    b = (1-(np.sum(class_vector)/len(class_vector)))

    imp = np.add(a*b, a*b)
    return imp

    


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    curr_impurity = 0.0
    prev_impurity = gini_impurity(previous_classes)
    tc = len(previous_classes)

    for c in current_classes:
        c_imp = gini_impurity(c)
        res = len(c)/tc * c_imp
        curr_impurity = curr_impurity+res
    
    g_gain = prev_impurity - curr_impurity
    return g_gain




class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth = float('inf')):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        max_a = 0
        opt_feat = 0
        holder1 = 0
        holder2 = 0
        half_len = len(classes)/2
        sum_classes = np.sum(classes)

        if depth <= holder1:
            if sum_classes<=half_len:
                return DecisionNode(None, None, None, 0)
            elif sum_classes>half_len:
                return DecisionNode(None, None, None, 1)
            
     
        else: 
            num_of_cols = features.shape[1]
            for i in range(num_of_cols):
                data = features[:, i]
                sum = np.sum(data)
                tc_elements = np.size(data)
                average = sum/tc_elements

                if np.max(data) - np.min(data) != 0:
                    a = gini_gain(classes, [data[data < average], data[data >= average]])

                    if max(a, max_a) == a:
                        opt_feat = i
                        max_a = a
            

            parsed_features_0 = features[0]

            
            if np.all(parsed_features_0==features):
                if sum_classes <= half_len:
                    return DecisionNode(None, None, None, 0)  
                elif sum_classes>half_len:
                    return DecisionNode(None, None, None, 1)
                
            equality = np.max(classes)-np.min(classes)
                
            if equality == 0:
                return DecisionNode(None, None, None, classes[0])
        

            sum = np.sum(features[:, opt_feat])
            tc = np.size(features[:, opt_feat])
            average = sum/tc

            #checking left and right indices and finding location
            right = np.where(features[:, opt_feat] >= average)[0]
            left = np.where(features[:, opt_feat] < average)[0]


        decision_tree_root = DecisionNode(None, None, lambda a : average > a[opt_feat])
        #recursively building tree method
        decision_tree_root.right = self.__build_tree__(features[right, :], classes[right], depth - 1)
        decision_tree_root.left = self.__build_tree__(features[left, :], classes[left], depth - 1)


        return decision_tree_root

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        r = self.root
        labels = []
        for k in range(len(features)):
            features_to_select = features[k, :]
            labels = np.concatenate((labels, [r.decide(features_to_select)]))

        return labels

class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit

        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        x = 1
        y = 0
        temp = []
        e_rate = self.example_subsample_rate
        tc_trees = self.num_trees
        a_rate = self.attr_subsample_rate

        decision_tree = DecisionTree()

        len = tc_trees

        f = (int(features.shape[1]) * a_rate)
        size_of_f = int(f)
     
        e = (int(features.shape[0]) * e_rate)
        size_of_e = int(e)
        
        
        while len > 0:
            c = np.random.permutation(int(features.shape[1]))[:size_of_f]
            f = np.random.permutation(int(features.shape[0]))[:size_of_e]

            sample_c =  classes[f]    
            sample_f = features[f,:][:,c]
            
            sample_tuple = (sample_f, sample_c)

            self.trees.append((decision_tree.__build_tree__(sample_f, sample_c), c))
            len-=1
        
    
            


    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        x = 0
        y = 1 #for classification purposes

        labels = [] 
        choices = []
        tc_trees = self.trees
        e_rate = self.example_subsample_rate
        a_rate = self.example_subsample_rate

        for i in range(features.shape[0]):
            decisions = []

            for tree in tc_trees:
                rn, c = tree
                
                particular_data = features[:, c]

                particular_choice = rn.decide(particular_data[i])
                
                decisions = np.concatenate((decisions, [particular_choice])) 

                #labels = np.concatenate((labels, [decision]))

            mean = np.mean(decisions)
            if mean <= 1/2:
                labels += [x]
            elif mean > 1/2:
                labels += [y]
    
        return labels



class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        mult = np.multiply(data, data)

        vectorized = np.add(mult, data)
        return vectorized

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        #organizing sums of rows
        max_row_sum = np.sum(data[:100], axis=1)
        #finding index of max row
        max_index = np.argmax(max_row_sum)
        #returning vectorized slice
        slice = (max_row_sum[max_index], max_index)

        return slice

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        data = data[data>0]
        data, occurrences = np.unique(data, return_counts=True)
        pos = dict(zip(data, occurrences)) 
        list = pos.items()
        return list

    
    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
            
        """

        if dimension == 'c':
            return np.insert(data, data.shape[1], vector, axis=1)
        else:
            return np.insert(data, data.shape[0], vector, axis=0)
         

        


    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """


        binary_mask = data < threshold
        res = np.where(binary_mask, np.square(data), data)
        return res

def return_your_name():
    return 'Rishab Soman'
