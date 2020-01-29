

import pandas as pd
import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 1) I used 10-fold Cross Validation on training set to test differnt methods, features and models. This file is for the model with highest accuracy on the 10-fold CV.

# 2) This is just the standard format with training and test split, i.e. train the Logistic Regression model on the training set and predict on the test set, then submit the predictions to Kaggle to see the result.

# 3) This model clean the text by down casing and removing punctuations, it uses unigram and bigrams as text features with TF-IDF representations, and no numerical features are used. The settings of the Logistic Regression model are as default in the sklearn package.


# read training file
df_train = pd.read_csv("train.tsv")

# the parent comment and the empty comment will not be used
df_train = df_train[["label", "comment"]]
df_train.dropna(inplace=True)


def CleanText(raw_comment):
    # 1. lower case
    new_comment = raw_comment.lower()
    # 2. remove punctuation
    new_comment = re.sub(r"[^\w\s]", "", new_comment)
    return new_comment


# clean comment by down casing and remove punctuations
for index, row in df_train.iterrows():
    raw_comment = df_train.iloc[index, "comment"]
    df_train.iloc[index, "comment"] = CleanText(raw_comment)


# construct corpus
list_all_words = []
for i in df_train.comment:
    words = word_tokenize(i)
    for word in words:
        list_all_words.append(word)



# construct TF-IDF matrix for training data
tfidf_vectorizer = TfidfVectorizer(input=list_all_words, lowercase=True, min_df=2, ngram_range=(1, 2))
tfidf_matrix_train = tfidf_vectorizer.fit_transform(df_train.comment)
feature_names = tfidf_vectorizer.get_feature_names()
df_tfidf_train = pd.DataFrame(tfidf_matrix_train.toarray(), columns=feature_names)

# construct labels
target_list = []
for i in df_train.label:
    target_list.append(i)
    #array list
y = pd.Series(target_list)

# train Logistic Regression model
logmodel = LogisticRegression()
logmodel.fit(df_tfidf_train, y)

# read test file
df_test = pd.read_csv("test.tsv")
df_test = df_test[["id", "comment"]]
df_test.fillna("", inplace=True)

# construct TF-IDF matrix for test data
tfidf_matrix_test = tfidf_vectorizer.transform(df_test.comment)
df_tfidf_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=feature_names)

# predict
y_pred = logmodel.predict(df_tfidf_test)
pd.Series(y_pred).to_csv("LR.csv", header=["label"], index_label="id")
