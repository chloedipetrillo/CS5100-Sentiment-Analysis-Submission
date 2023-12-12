import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import time
import os

# our import
from data_preprocessing import *

class BagOfWordsNGramVectorizer():
    '''
    Class BagofWordsNGramVectorizer is a class that vectorizes data using the
        bag of words vectorization technique with ngram incoorporation.
    '''
    def __init__(self, n):
        # initialize variables
        self.term_index = {}
        self.vector_size = 0
        self.n = n
    
    def establish_term_indexes(self, terms):
        '''
        Method - establish_term_indexes: takes in a Counter of all terms and assigns each
            unqiue term a unique index which will correspond with its column in the 
            vectorizaiton process.
        Parameters:
            vocab: (Counter) terms as keys, frequency as value
        Returns: None
        '''
        i = 0
        for word in terms.keys():
            self.term_index[word] = i
            i+=1

    def transform(self, data, training=False):
        '''
        Method - transform: takes in review data and will return a vectorized, numerical representation
            of the data
        Parameters:
            data: pandas series containing review data
            training: (boolean) whether or not the data is training data
        Returns: NumPy array with vectorized data
        '''
        # if you are in training, establish the unique vocabulary and each words index
        if training:
            total_vocab = Counter()
            for review in data:
                ngrams = self.split_words(review)

                total_vocab.update(ngrams)
            
            self.establish_term_indexes(total_vocab)
            self.vector_size = len(total_vocab.keys())

        # vectorize data based on term counts using predetermined term indexes from training
        all_vectors = []
        for review in data:
            review_vector = [0] * self.vector_size

            for term in self.split_words(review):
                if term in self.term_index:
                    review_vector[self.term_index[term]] +=1
            
            all_vectors.append(review_vector)

        return np.array(all_vectors)

    def split_words(self, review):
        '''
        Method - split_words: creates n-grams from a review by taking in review splitting the words
            into terms depending on the ngram factor
        Parameters:
            review: (string) review text
        Returns:
            ngram: list of n-grams generated from the review
        '''
        review = review.split(" ")
        ngram = []
        if len(review) >= self.n:
            for i in range(len(review) - (self.n -1)):
                term = ''
                for j in range(self.n):
                    term += review[i+j] + ' '
                ngram.append(term.strip())

        return ngram


def run_svm(filename, kernel, test_size, data_size, ngram=1): 
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
    print(f"Running {python_file_name} with a {kernel} kernel on file {filename} with data size {data_size} ngram {ngram} and using a test size of {test_size}!")

    start_time = time.time()

    # get the data
    data = get_clean_data(filename, data_size, lemm=True, stem=True)
    train_data, test_data, train_labels, test_labels = split_data(test_size, data)

    # vectorize the data
    vectorizer = BagOfWordsNGramVectorizer(ngram)
    train_vectorized = vectorizer.transform(train_data, True)
    test_vectorized = vectorizer.transform(test_data)

    # svm
    svm = SVC(kernel=kernel,tol=1e-3)
    svm.fit(train_vectorized, train_labels)
    test_predictions = svm.predict(test_vectorized)

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