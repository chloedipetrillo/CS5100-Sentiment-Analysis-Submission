# imports
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter
import os
import math

# our own imports
from data_preprocessing import *

def train(train_data, train_labels):
    '''
    Function - train: takes parameters train_data and train labels. Using Naive Bayes, the probability
        of each label is calculated, and the probabilities of features given the label.
    Parameters:
        train_data: pandas series containing review data
        train_labels: pandas series containing sentiment labels
    Returns:
        prob_pos: (float) probability of the label -- probability of a positivie sentiment 
        prob_neu: (float) probability of the label -- probability of a neutral sentiment 
        prob_neg: (float) probability of the label -- probability of a negative sentiment 
        prob_word_given_positive: (Counter) counter containing probability of each feature given the label 
            -- probability of word given positive sentiment
        prob_word_given_negative: (Counter) counter containing probability of each feature given the label 
            -- probability of word given negative sentiment
        prob_word_given_neutral: (Counter) counter containing probability of each feature given the label 
            -- probability of word given neutral sentiment
        positive_word_count: (int) number of words in the positive class training set
        word_count_negative: (int) number of words in the negative class training set
        word_count_neutral: (int) number of words in the neutral class training set
        total_vocabulary_size: (int) total unique words in the vocabulary across all classes 
    '''

    # initialize variables 
    total_vocabulary = set() 
    positive_word_frequency = Counter()
    negative = Counter()
    neutral = Counter()
    reviews_pos = 0
    positive_word_count = 0
    reviews_neg = 0 
    word_count_negative = 0
    reviews_neu = 0 
    word_count_neutral = 0

    # calculate word frequencies per class and total vocabulary
    for review, sentiment in zip(train_data.values, train_labels):

        if sentiment == "Positive":
            reviews_pos +=1
            for word in review.split():
                positive_word_frequency[word] +=1
                positive_word_count+=1
                total_vocabulary.add(word)
        elif sentiment == "Neutral":
            reviews_neu +=1
            for word in review.split():
                neutral[word] +=1
                word_count_neutral+=1
                total_vocabulary.add(word)
        
        elif sentiment == "Negative":
            reviews_neg +=1
            for word in review.split():
                negative[word] +=1
                word_count_negative+=1
                total_vocabulary.add(word)

    total_vocabulary_size = len(total_vocabulary)

    # probability of the label
    prob_pos = reviews_pos/(reviews_pos+reviews_neg+reviews_neu)
    prob_neu = reviews_neu/(reviews_pos+reviews_neg+reviews_neu)
    prob_neg = reviews_neg/(reviews_pos+reviews_neg+reviews_neu)

    # initialize Counters for probabilities of feature given the label
    prob_word_given_positive = Counter()
    prob_word_given_negative = Counter()
    prob_word_given_neutral = Counter()

    # calculate the probabilities of the feature given the label using smoothing
    for word in total_vocabulary:
        prob = (positive_word_frequency[word] +1 ) / (positive_word_count + total_vocabulary_size)
        prob_word_given_positive[word] = prob
        prob = (negative[word] +1 ) / (word_count_negative + total_vocabulary_size)
        prob_word_given_negative[word] = prob
        prob = (neutral[word] +1 ) / (word_count_neutral + total_vocabulary_size)
        prob_word_given_neutral[word] = prob


    return (prob_pos, prob_neu, prob_neg, prob_word_given_positive, prob_word_given_negative, 
            prob_word_given_neutral, positive_word_count, total_vocabulary_size, word_count_negative, word_count_neutral)





def test(test_data, test_labels, prob_pos, prob_neu, prob_neg, prob_word_given_positive, prob_word_given_negative, 
         prob_word_given_neutral, positive_word_count, total_vocabulary_size, word_count_negative, word_count_neutral):
    '''
    Function - test: Using the probabilities calculated for the labels and the features given the label learned 
        during training, test will go through the test_data and make classifications of each review by determining
        the probability of a review given a class using log probabilities. Once all documents have been 
        classified, accuracy and a classification report will be printed.
    Parameters:
        test_data: pandas series containing test review data
        test_labels: pandas series containing test sentiment labels
        prob_pos: (float) probability of the label -- probability of a positivie sentiment 
        prob_neu: (float) probability of the label -- probability of a neutral sentiment 
        prob_neg: (float) probability of the label -- probability of a negative sentiment 
        prob_word_given_positive: (Counter) counter containing probability of each feature given the label 
            -- probability of word given positive sentiment
        prob_word_given_negative: (Counter) counter containing probability of each feature given the label 
            -- probability of word given negative sentiment
        prob_word_given_neutral: (Counter) counter containing probability of each feature given the label 
            -- probability of word given neutral sentiment
        positive_word_count: (int) number of words in the positive class training set
        word_count_negative: (int) number of words in the negative class training set
        word_count_neutral: (int) number of words in the neutral class training set
        total_vocabulary_size: (int) total unique words in the vocabulary across all classes 
    Returns:
        accuracy: (float) accuracy of the naive bayes model
    '''

    # initialize variables
    total_tests = 0
    right = 0
    wrong = 0
    predictions = []

    # for each test review, classify its sentiment using the trained model
    for review, sentiment in zip(test_data.values, test_labels):
        # probability of the class
        pos = math.log(prob_pos)
        neg = math.log(prob_neu)
        neu = math.log(prob_neg)

        # using log probabilities (so adding probabilities) determine probability of a class given a document
        for  word in review.split():
            # laplace smoothing -- adds default if word has never been seen before
            pos += math.log(prob_word_given_positive.get(word, (1 / (positive_word_count + total_vocabulary_size))))
            neu += math.log(prob_word_given_neutral.get(word, 1 / (word_count_neutral + total_vocabulary_size)))
            neg += math.log(prob_word_given_negative.get(word, 1 / (word_count_negative + total_vocabulary_size)))
        
        # determine the classification prediction and whether or not it was right
        vals = [[pos, "Positive"], [neu, "Neutral"],[neg, "Negative"]]
        prediction = max(vals)
        predictions.append(prediction[1])
        if prediction[1] == sentiment:
            right +=1
        else:
            wrong +=1
        total_tests +=1

    predictions = np.array(predictions)

    # print results
    print("Right: ", right)
    print("Wrong: ", wrong)
    print("Accuracy: ", (right/total_tests))
    print("Naive Bayes Accuracy:", accuracy_score(test_labels, predictions))
    print("Naive Bayes Classification Report:\n", classification_report(test_labels, predictions, zero_division=1))

    return (right/total_tests)

def run_naive_bayes(filename, test_size, data_size):
    '''
    Function - run_naive_bayes: acts as a driver function to run Naive Bayes. Given a data set filename, a test_size, and a 
        data_size, run_naive_bayes will split the data into train and test data, train the model using the train data, then
        test the test data, ultimately returning the model's accuracy.
    Parameters:
        filename: (string) name of the file for the dataset
        test_size: (float) the decimal size of the test size
        data_size: (int) number of rows of data to use
    Returns: (float) accuracy of the model
    '''
    python_file_path = __file__
    python_file_name = os.path.basename(python_file_path)
    print(f"Running {python_file_name} on file {filename} with data size {data_size} using a test size of {test_size}!")
    
    # get the data
    data = get_clean_data(filename, data_size, lemm=False, stem=False)
    train_data, test_data, train_labels, test_labels = split_data(test_size, data)

    # train
    (prob_pos, prob_neu, prob_neg, prob_word_given_positive, 
     prob_word_given_negative, prob_word_given_neutral, positive_word_count, 
     total_vocabulary_size, word_count_negative, word_count_neutral) = train(train_data, train_labels)
    
    # test
    accuracy  = test(test_data, test_labels, prob_pos, prob_neu, prob_neg, prob_word_given_positive, 
     prob_word_given_negative, prob_word_given_neutral, positive_word_count, total_vocabulary_size, word_count_negative, word_count_neutral)
    
    return accuracy
