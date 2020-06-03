# StakeOverFlow-Tag-Prediction
Predicting StakeOverFlow with Linear Model
In this notebook we will be predicting tags for post in StakeOverflow and will be using multilabel classifier approch.

1. **Importing the Python Librarie**
- Numpy — a package for scientific computing.
- Pandas — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
- scikit-learn — a tool for data mining and data analysis.
- NLTK — a platform to work with natural language.

2. **Text Processing**
In this task you will deal with a dataset of post titles from StackOverflow. You are provided a split to 3 sets: train, validation and test. All corpora (except for test) contain titles of the posts and corresponding tags (100 tags are available). 

3. **Convert our text data into Vector format**
Machine Learning algorithms work with numeric data and we cannot use the provided text data "as is". There are many ways to transform text data to numeric vectors. In this task you will try to use two of them.

**Bag of words**
One of the well-known approaches is a bag-of-words representation. To create this transformation, follow the steps:

Find N most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
For each title in the corpora create a zero vector with the dimension equals to N.
For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.
Let's try to do it for a toy example. Imagine that we have N = 4 and the list of the most popular words is

['hi', 'you', 'me', 'are']

Then we need to numerate them, for example, like this:

{'hi': 0, 'you': 1, 'me': 2, 'are': 3}

And we have the text, which we want to transform to the vector:

'hi how are you'

For this text we create a corresponding zero vector

[0, 0, 0, 0]

And interate over all words, and if the word is in the dictionary, we increase the value of the corresponding position in the vector:

'hi':  [1, 0, 0, 0]
'how': [1, 0, 0, 0] # word 'how' is not in our dictionary
'are': [1, 0, 0, 1]
'you': [1, 1, 0, 1]
Transforming text to a vector
Machine Learning algorithms work with numeric data and we cannot use the provided text data "as is". There are many ways to transform text data to numeric vectors. In this task you will try to use two of them.

Bag of words
One of the well-known approaches is a bag-of-words representation. To create this transformation, follow the steps:

Find N most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
For each title in the corpora create a zero vector with the dimension equals to N.
For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.
Let's try to do it for a toy example. Imagine that we have N = 4 and the list of the most popular words is

['hi', 'you', 'me', 'are']

Then we need to numerate them, for example, like this:

{'hi': 0, 'you': 1, 'me': 2, 'are': 3}

And we have the text, which we want to transform to the vector:

'hi how are you'

For this text we create a corresponding zero vector

[0, 0, 0, 0]

And interate over all words, and if the word is in the dictionary, we increase the value of the corresponding position in the vector:

'hi':  [1, 0, 0, 0]
'how': [1, 0, 0, 0]

word 'how' is not in our dictionary
'are': [1, 0, 0, 1]
'you': [1, 1, 0, 1]

**TF-IDF**
The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora. It helps to penalize too frequent words and provide better features space.

Implement function tfidf_features using class TfidfVectorizer from scikit-learn. Use train corpus to train a vectorizer. Don't forget to take a look into the arguments that you can pass to it. We suggest that you filter out too rare words (occur less than in 5 titles) and too frequent words (occur more than in 90% of the titles). Also, use bigrams along with unigrams in your vocabulary.

4. **Now classifying the Multilabels**
To deal with such kind of prediction, we need to transform labels in a binary form and the prediction will be a mask of 0s and 1s. For this purpose it is convenient to use MultiLabelBinarizer from sklearn.
Implement the function train_classifier for training a classifier. In this task we suggest to use One-vs-Rest approach, which is implemented in OneVsRestClassifier class. In this approach k classifiers (= number of tags) are trained. As a basic classifier, use LogisticRegression. It is one of the simplest methods, but often it performs good enough in text classification tasks. It might take some time, because a number of classifiers to train is large.

5. **Evaluation of Result**
To evaluate the results we will use several classification metrics:

**Accuracy**
- F1-score
- Area under ROC-curve
- Area under precision-recall curve
