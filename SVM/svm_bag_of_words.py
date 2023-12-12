import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import time
import os

# our import
from data_preprocessing import *

class BagOfWordsVectorizer():
    '''
    Class BagofWordsVectorizer is a class that vectorizes data using the
        bag of words vectorization technique.
    '''
    def __init__(self):
        # initialize variables
        self.word_index = {}
        self.vector_size = 0
    
    def establish_word_indexes(self, vocab):
        '''
        Method - establish_word_indexes: takes in a Counter of all vocab words and assigns each
            unqiue vocab word a unique index which will correspond with its column in the 
            vectorizaiton process.
        Parameters:
            vocab: (Counter) vocabulary words as keys, frequency as value
        Returns: None
        '''
        i = 0
        for word in vocab.keys():
            self.word_index[word] = i
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
                individual_words = review.split()
                total_vocab.update(individual_words)
            
            self.establish_word_indexes(total_vocab)
            self.vector_size = len(total_vocab.keys())
            

        # vectorize data based on word counts using predetermined word indexes from training
        all_vectors = []
        for review in data:
            review_vector = [0] * self.vector_size
            # calculate word frequencies per review
            for word in review.split():
                if word in self.word_index:
                    review_vector[self.word_index[word]] +=1
            
            all_vectors.append(review_vector)

        return np.array(all_vectors)



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

    # set up data
    data = get_clean_data(filename, data_size, lemm=True, stem=True)
    train_data, test_data, train_labels, test_labels = split_data(test_size, data)

    # vectorize data
    vectorizer = BagOfWordsVectorizer()
    train_vectorized = vectorizer.transform(train_data, True)
    test_vectorized = vectorizer.transform(test_data)

    # run svm
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
