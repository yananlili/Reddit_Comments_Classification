# Reddit_Comments_Classification

# Task
This is a text classification task. Every document (a line in the data file) contains a Reddit comment and its parent post (the post being commented on), separated with tab.

Your goal is to classify each Reddit comment (not the parent post) into ONE of the two categories, based on whether it is sarcastic or not. The parent post provides necessary context for your judgment.

0: the comment does NOT express sarcasm

1: the comment expresses sarcasm

The training data contains 53,032 comments, already labeled with one of the above two categories. The test data contains 17,719 comments that are unlabeled. The submission should be a .csv (comma separated free text) file with a header line "id,label" followed by exactly 17,719 lines. In each line, there should be exactly two integers, separated by a comma. The first integer is the line ID of a test example (0, 1, 2, ..., 17718), and the second integer is the category (or label) predicted by your classifier, one of {0,1}.

# classification_keras
Using keras, CNN, RNN

# classification_sklearn
Using nltk and sklearn, performing better
1) I used 10-fold Cross Validation on training set to test differnt methods, features and models. This file is for the model with highest accuracy on the 10-fold CV.

2) This is just the standard format with training and test split, i.e. train the Logistic Regression model on the training set and predict on the test set, then submit the predictions to Kaggle to see the result.

3) This model clean the text by down casing and removing punctuations, it uses unigram and bigrams as text features with TF-IDF representations, and no numerical features are used. The settings of the Logistic Regression model are as default in the sklearn package.
