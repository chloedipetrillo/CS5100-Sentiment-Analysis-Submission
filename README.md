Neha Annigeri & Chloe DiPetrillo<br>
Final Project <br>
CS 5100, Fall 2023

# Sentiment Analysis of Reviews

### Formal Project Description

To perform sentiment analysis on a labeled dataset, the formal definition of our project is proposed as:

We have documents (D) where (D) is a set of individual reviews: ( D = {r_1, r_2, ..., r_n} ) and classes (C) where ( C = {positive, negative, neutral} ).

We are given a training set ( {(r_i, c_i) | i=0...N} ), given a labeled set of input-output pairs ( D={(r_i,c_i)}_i ).

We want to find a good approximation of ( f: R -> C ).


### Project Summary
We utilized Naive Bayes and Support Vector Machines. We ran both of these algorithms with variations of different parameters such as training-testing split, data, featurization techniques, document length, and vectorization techniques.


### Running Instructions
We made a basic command line interface because we were dealing with a lot of files and experiment iterations. To run our code follow the instructions listed below:
Run the following command.
```python driver.py ```
After, you will be prompted with a series of questions. Follow the directions listed to choose the parameters for the supervised learning models. 



