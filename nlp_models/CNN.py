import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn import metrics

def loadData_Tokenizer(X_train, X_test, MAX_NB_WORDS=75000, MAX_SEQUENCE_LENGTH=1000):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizer.word_index
    return X_train, X_test, word_index

def Build_Model_CNN_Text(word_index, nclasses, MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=100, dropout=0.5):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nclasses, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# Load datasets
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Preprocess data
MAX_SEQUENCE_LENGTH = 500
X_train, X_test, word_index = loadData_Tokenizer(X_train, X_test, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

# Build and train the model
model_CNN = Build_Model_CNN_Text(word_index, 20, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

model_CNN.summary()

model_CNN.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=20,
              batch_size=64,
              verbose=2)

# Evaluate the model
predicted = model_CNN.predict(X_test)
predicted = np.argmax(predicted, axis=1)

model_CNN.save('../trained_models/20News_CNN.keras')

# Print classification report
print(metrics.classification_report(y_test, predicted))
