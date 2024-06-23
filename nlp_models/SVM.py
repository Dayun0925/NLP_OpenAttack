## Code from https://github.com/kk7nc/Text_Classification/blob/master/code/SVM.py

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
import pickle

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(probability= True)),
                     ])

text_clf.fit(X_train, y_train)

with open("../trained_models/20News_SVM_probs.pkl","wb") as f:
    pickle.dump(text_clf,f)

predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))

## NOTE
## output type: a label per sample without a confidence vector