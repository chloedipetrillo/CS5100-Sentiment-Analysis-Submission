import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
import string
import os



import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("wordnet")



def lemmatize_text(review):
    '''
    Function - lemmatize_text: takes in a text review as a string and will return the review after
        performing lemmatization
    Parameters:
        review: (string) the review
    Returns: the string with the text lemmatized
    '''
    tokenizer = WhitespaceTokenizer()
    lem = nltk.stem.WordNetLemmatizer()
    lemmatized_review = ""
    for word in tokenizer.tokenize(review):
        lemmatized_review += lem.lemmatize(word) + " "
    return lemmatized_review

def stem_text(review):
    '''
    Function - stem_text: takes in a text review as a string and will return the review after
        performing stemming
    Parameters:
        review: (string) the review
    Returns: the string with the text stemmed
    '''
    stemmer = PorterStemmer()
    tokenizer = WhitespaceTokenizer()
    stem = []
    for word in tokenizer.tokenize(review):
        stem.append(stemmer.stem(word))
    return ' '.join(stem)
 
def remove_punctuation(review):
    '''
    Function - remove_punctuation: takes in a text review as a string and will return the review after
        removing all punctuation
    Parameters:
        review: (string) the review
    Returns: the string with the punctuation removed
    '''
    # replace punctuation with nothing
    trans = str.maketrans('', '', string.punctuation)
    return review.translate(trans)

def remove_stop_words(review):
    '''
    Function - remove_stop_words: takes in a text review as a string and will return the review after
        removing all stop words
    Parameters:
        review: (string) the review
    Returns: the string without stop words
    '''
    stop_words = set(stopwords.words('english'))
    good_words = []
    for word in str(review).split():
        if word not in stop_words:
            good_words.append(word)
    return ' '.join(good_words)


def get_clean_data(filename, data_size="all", lemm=True, stem=True):
    '''
    Function - get_clean_data: takes a file name, a data size and flags for whether or not 
        to include stemming/lemmatization of the text and reads in the data from the specified
        filename and will process data. 
    Parameters:
        filename: (string) the file name for data set using
        data_size: (int) number of rows of data to use (default is all for entire data set)
        lemm: (boolean) whether or not to include lemmatization (default True)
        stem: (boolean) whether or not to include stemming (default True)
    Returns: the string with the text lemmatized
    '''
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", filename)
    data = pd.read_csv(file_path)
    if data_size != "all":
        data = data.sample(n=data_size, random_state=42).reset_index(drop=True)
    
    data['review'] = data.review.apply(remove_stop_words)

    if lemm:
        data['review'] = data.review.apply(lemmatize_text)
    if stem:
        data['review'] = data.review.apply(stem_text)
    
    data['review'] = data.review.apply(remove_punctuation)
    
    data['sentiment'] = data['rating'].map({1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'})

    return data

def split_data(test_size, data):
    '''
    Function - split_data: takes in a float for test size and the full data set and
        splits the data into training and test sets with their respective labels
    Parameters:
        test_size: (float) the decimal size of the test size
        data: all the data as pandas data frame
    Returns: the train data, train labels, test data, and test labels all as pandas series
    '''
    train_data, test_data, train_labels, test_labels = train_test_split(
        data['review'], data['sentiment'], test_size=test_size, random_state=42)

    return train_data, test_data, train_labels, test_labels