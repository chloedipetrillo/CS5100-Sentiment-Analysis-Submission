import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import time
import os

# our import
from data_preprocessing import *


class TFIDF():
    '''
    Class TFIDF is a class that vectorizes data using the
        term frequency - inverse document frequency technique.
    '''
    def __init__(self):
        # initialize variables
        self.tf_values  = []
        self.idf_values = {}
        self.vocab = []
        

    def get_tf_values(self, data):
        '''
        Method - get_tf_values: given data, this method will establish all the tf_values 
        Parameters:
            data: pandas series representing all the reviews
        Returns: None
        '''

        # remove punctuation
        data = [remove_punctuation(text) for text in data]

        # for each row (review) append a counter for where the key is a word and value is tf value
        for review in data:
            words = review.split()
            word_count_per_review = Counter(words)
            num_words = len(words)
            tf_vector = {}
            for word, frequency in word_count_per_review.items():
                tf_val = 0  
                if num_words > 0:
                    tf_val = frequency / num_words
                tf_vector[word] = tf_val
            self.tf_values.append(tf_vector)
    
    def get_idf_values(self, data):
        '''
        Method - get_idf_values: given data, this method will establish all the idf_values 
        Parameters:
            data: pandas series representing all the reviews for training data
        Reutrns: None
        '''

        # calculate the idf value where each word is a key and value is idf value
        idf_matrix = {}
        for review in data:
            words = review.split()
            words = set(words)
            # incrementing document frequency per word
            for word in words:
                idf_matrix[word] = idf_matrix.get(word, 0)+1
        num_reviews = len(data)
        # use smoothing
        for word, frequency in idf_matrix.items():
            self.idf_values[word] = np.log((num_reviews +1) / (frequency + 1)) +1
        
        # establish a list of vocabulary
        self.vocab = list(self.idf_values.keys())

    def train_tfidf_values(self):
        '''
        Method - train_tfidf_values: generates tfidf values during training 
        Parameters: None
        Reutrns: 
            tfidf_train: NumPy array matrix of all zeroes of appropriate size
            tfidf_values: array of dictionaries where each dictionary corresponds to a review
                and the key in each dictionary is a word in that review and each value is the
                tfidf value for that word in that review
        '''
        tfidf_values = []

        # adds a dictionary for each review to the array where each dictionary key is a word in
        # that review and each value is that words tfidf value for that review
        for tf_vector in self.tf_values:
            vector = {}
            for word, tf in tf_vector.items():
                vector[word] = tf * self.idf_values[word]
            tfidf_values.append(vector)

        # creates the matrix of appropriate size based on num training reviews and num vocab words
        tfidf_train = np.zeros((len(tfidf_values), len(self.vocab)))

        return tfidf_train, tfidf_values
    
    def test_tfidf_values(self, data):
        '''
        Method - test_tfidf_values: generates tfidf values for test data
        Parameters:
            data: pandas series representing all the reviews for test data
        Returns:
            tfidf_test: NumPy array matrix of all zeroes of appropriate size
            tfidf_values: array of dictionaries where each dictionary corresponds to a review
                and the key in each dictionary is a word in that review and each value is the
                tfidf value for that word in that review
        '''
        data = [remove_punctuation(text) for text in data]

        tfidf_values = []

        for review in data:
            # use a dictionary to keep track of the tfidf value for each word per review
            vector = {}
            words = review.split()
            # establish word frequency per review
            word_count_per_review = Counter(words)
            num_words = len(words)
            
            # for each word calculate the tfidf value
            for word in self.vocab:
                tf_val = 0  
                if num_words > 0:
                    tf_val = word_count_per_review.get(word, 0) / num_words
                idf_val = self.idf_values.get(word, 0)
                tfidf_val = tf_val * idf_val 
                vector[word] = tfidf_val

            # add dictionary to the array corresponding with each review
            tfidf_values.append(vector)

        tfidf_test = np.zeros((len(tfidf_values), len(self.vocab)))
        return tfidf_test, tfidf_values


    def transform(self, tfidf_and_matrix):
        '''
        Method - transform: transforms the given matrix using precomputed tfidf values
        Parameters:
            tfidf_and_matrix: Tuple containing the matrix used to fill in vectorized data in 
            an established order and tfidf values stored in dictionaries for each review
        Returns:
            tfidf: NumPy array with tfidf values
        '''
        tfidf, matrix = tfidf_and_matrix

        for i in range(len(matrix)):
            vector = matrix[i]

            for j in range(len(self.vocab)): # for a consistent order
                word = self.vocab[j]

                # get the tfidf value for that word in that review
                tfidf_value = vector.get(word, 0)

                # fill values in the tfidf matrix
                tfidf[i, j] = tfidf_value

        return tfidf
    
    def train_transform(self, data):
        '''
        Method - train_transform: vectorizes train data
        Parameters:
            data: pandas series representing all the reviews for train data
        Returns:
            tfidf: NumPy array with tfidf values
        '''
        self.get_idf_values(data)
        self.get_tf_values(data)
        return self.transform(self.train_tfidf_values())
    
    def test_transform(self, data):
        '''
        Method - test_transform: vectorizes test data
        Parameters:
            data: pandas series representing all the reviews for test data
        Returns:
            tfidf: NumPy array with tfidf values
        '''
        return self.transform(self.test_tfidf_values(data))


def run_svm(filename, kernel, test_size, data_size):
    '''
    Method - run_svm: takes in filename for the data set being used, a kernel for SVM, a test size, and a data size, 
        and runs svm.
    Parameters:
        filename: (string) name of the file for the dataset
        kernel: (string) for the kernel being used with SVM
        test_size: (float) the decimal size of the test size
        data_size: (int) number of rows of data to use
    Returns: (float) accuracy of the model
    '''
    python_file_path = __file__
    python_file_name = os.path.basename(python_file_path)
    print(f"Running {python_file_name} with a {kernel} kernel on file {filename} with data size {data_size} using a test size of {test_size}!")
    start_time = time.time()

    # get the data - lem and stem by default are true
    data = get_clean_data(filename, data_size)
    train_data, test_data, train_labels, test_labels = split_data(test_size, data)

    # vectorize data
    vectorizer = TFIDF()
    train_vectors = vectorizer.train_transform(train_data)
    test_vectors = vectorizer.test_transform(test_data)

    # run svm
    svm_classifier = SVC(kernel=kernel)
    svm_classifier.fit(train_vectors, train_labels)
    test_predictions = svm_classifier.predict(test_vectors)

    # print results
    accuracy = accuracy_score(test_labels, test_predictions)
    print("Accuracy:", accuracy)
    report = classification_report(test_labels, test_predictions, zero_division=1)
    print("Classification Report:\n", report)


    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("--- %dh %dm %ds ---" % (hours, minutes, seconds))
    return accuracy