import os
from importlib import util


def run():
    '''
    Function - run: main driver for the command line interface. Runs program
    Parameters: None
    Returns: None
    '''
    algo = input("\nWhich algorithm do you want to run?\n"+
                 "A. Naive Bayes\nB. SVM\n").lower()
    
    while algo not in ["a", "b"]:
        algo = input("\nThat is an invalid selection. Please only enter a letter.\nWhich algorithm do you want to run?\n"+
                 "A. Naive Bayes\nB. SVM\n").lower()

    if algo == "a":
        run_nb()
    else:
        svm()


def svm():
    '''
    Function - svm: driver for running svm algorithm
    Parameters: None
    Returns: None
    '''
    filename = determine_svm_version()
    if "ngram" in filename:
        n = ngram()
    kernel = determine_kernel()
    data_file = determine_filename()
    size = determine_datasize(data_file)
    test = determine_test_size()
    print()

    # use the name of the file to get that specific svm function and run it
    current_directory = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(current_directory, "SVM", filename+".py")

    s = util.spec_from_file_location(filename, path)
    m = util.module_from_spec(s)
    s.loader.exec_module(m)

    run_svm = getattr(m, "run_svm")

    if "ngram" in filename:
        run_svm(data_file, kernel, test, size, n)
    else:
        run_svm(data_file, kernel, test, size)


def run_nb():
    '''
    Function - run_nb: driver for running naive bayes algorithm
    Parameters: None
    Returns: None
    '''
    filename = determine_naive_bayes_version()
    file = determine_filename()
    size = determine_datasize(file)
    test = determine_test_size()
    print()

    # use the name of the file to get that specific naive bayes function and run it
    current_directory = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(current_directory, "NaiveBayes", filename+".py")

    s = util.spec_from_file_location(filename, path)
    m = util.module_from_spec(s)
    s.loader.exec_module(m)

    run_naive_bayes = getattr(m, "run_naive_bayes")

    run_naive_bayes(file, test, size)


def determine_kernel():
    '''
    Function - determine_kernel: takes user input to determine kernel used for svm
    Parameters: None
    Returns: string for kernel type
    '''
    k = input("\nWhich kernel would you like to use?\n"+
                 "A. linear\nB. polynomial\n").lower()
    while k not in ["a", "b"]:
        k = input("\nThat is an invalid selection. Please only enter a letter.\n\nWhich kernel would you like to use?\n"+
                 "A. linear\nB. polynomial\n").lower()
    
    if k == "a":
        return "linear"
    return "poly"



def determine_svm_version():
    '''
    Function - determine_svm_version: takes user input to determine vectorization techniques used in svm
    Parameters: None
    Returns: name of svm file you wish to run
    '''
    v = input("\nWhich vectorization technique do you want to use with SVM?\n"+
                 "A. Bag of Words\nB. TFIDF\nC. ngrams\n").lower()
    while v not in ["a", "b", "c"]:
        v = input("\nThat is an invalid selection. Please only enter a letter.\nWhich vectorization technique do you want to use with SVM?\n"+
                 "A. Bag of Words\nB. TFIDF\nC. ngrams\n").lower()
    
    if v == "a":
        s = "svm_bag_of_words"
    elif v == "b":
        y = input("\nWant to normalize? Type 'yes' or 'no'.\n").lower()
        if y == "yes":
            s = "svm_tfidf_normalize"
        else:
            s = "svm_tfidf"
    else:
        
        s = "svm_ngram"
    return s



def ngram():
    '''
    Function - ngram: takes user input to determine ngram value
    Parameters: None
    Returns: integer value for ngram factor
    '''
    return int(input("\nWhat value of n for ngram? Integers only. For example: 2.\n"))


def determine_naive_bayes_version():
    '''
    Function - determine_naive_bayes_version: takes user input to determine which naive bayes
        modifications to make
    Parameters: None
    Returns: name of naive bayes file you want to run
    '''
    nb = "nb"
    s = input("\nInclude stemming? Type 'yes' or 'no'.\n").lower()
    if s == "yes":
        nb += "_stem"

    l = input("\nInclude lemmatization? Type 'yes' or 'no'.\n").lower()
    if l == "yes":
        nb += "_lem"

    p = input("\nUse log probabilities? Type 'yes' or 'no'.\n").lower()
    if p == "yes":
        nb += "_log"

    return nb
        

def determine_filename():
    '''
    Function - determine_filename: takes user input to determine the filename of the data set
        modifications to make
    Parameters: None
    Returns: string name of dataset file name
    '''
    s = input("\nWhich data set do you want to use?\n"+
                 "A. Hotel Reviews (long reviews)\nB. Amazon Food Reviews (short reviews)\n").lower()
    
    while s not in ["a", "b"]:
        s = input("\nThat is an invalid selection. Please only enter a letter.\nWhich data set do you want to use?\n"+
                 "A. Hotel Reviews (long reviews)\nB. Amazon Food Reviews (short reviews)\n").lower()
      
    if s == "a":
        return "data.csv"
    return "summary_food_reviews.csv"

def determine_datasize(filename):
    '''
    Function - determine_datasize: takes user input to determine size of dataset user wants to use
    Parameters: 
        filename: string the file name of the data set
    Returns: 
        size of data set to use
    '''
    if filename == "data.csv":
        s = input("\nWhat size data set do you want to use? Type 'all' to use entire data set. \n"
                      +"Otherwise type a whole number less than 20000 indicating how many rows of data you would like to use.\n")
        try:
            s = int(s)
        except:
            s = str(s)
    else:
        s = input("\nWhat size data set do you want to use? Type 'all' to use entire data set. \n"
                      +"Otherwise type a whole number less than 568000 indicating how many rows of data you would like to use.\n")
        try:
            s = int(s)
        except:
            s = str(s)
        
    return s

def determine_test_size():
    '''
    Function - determine_test_size: takes user input to determine size of test size
    Parameters: None
    Returns: 
        float for test size
    '''

    try:
        s = float(input("\nWhat size test set do you want to use? \n"
                      +"Type percentage as a decimal. \nFor example: .9 is 10% training data, 90% test data\n"))
    except:
        print("A test size needs to be a number")

    
    return s





def main():
    try:
        run()
    except:
        print("error with selections")
       

if __name__ == "__main__":
    main()