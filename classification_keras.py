
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import os
os.environ['KERAS_BACKEND']='theano' # Why theano why not
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint


import csv
# %matplotlib inline



MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

df = pd.read_csv('train.tsv', sep='\t', header=0) 
df = df.dropna()
df = df.reset_index(drop=True)
print('Shape of dataset ',df.shape)
print(df.columns)
print('No. of unique classes',len(set(df['label'])))

macronum=sorted(set(df['label']))
macro_to_id = dict((note, number) for number, note in enumerate(macronum))

def fun(i):
    return macro_to_id[i]

df['label']=df['label'].apply(fun)

texts = []
labels = []


for idx in range(df.comment.shape[0]):
    text = BeautifulSoup(df.comment[idx])
    texts.append(clean_str(str(text.get_text().encode())))

for idx in df['label']:
    labels.append(idx)
    
#print(labels[:10])
#print(texts[:10])


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Number of Unique Tokens',len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of Data Tensor:', data.shape)
print('Shape of Label Tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]

labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]



embeddings_index = {}
f = open('glove.6B.100d.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

#Build a model
#l_cov1= Conv1D(200, 5, activation='relu')(embedded_sequences)
#l_pool1 = MaxPooling1D(2)(l_cov1)
#l_cov2 = Conv1D(200, 9, activation='relu')(l_pool1)
#l_pool2 = MaxPooling1D(2)(l_cov2)
#l_lstm = LSTM(300, return_sequences=True, activation="relu")(l_pool2)

#l_cov3 = Conv1D(300, 5, activation='relu')(l_lstm)
#l_pool3 = MaxPooling1D(16)(l_cov3)  # global max pooling
#l_flat = Flatten()(l_pool3)
#l_dense = Dense(400, activation='relu')(l_flat)
#preds = Dense(len(macronum), activation='softmax')(l_dense)


    
#
#model = Model(sequence_input, preds)
#model.compile(loss='categorical_crossentropy',
             # optimizer='rmsprop',
              #metrics=['acc'])

#print("Simplified convolutional neural network")
#model.summary()


#RNN
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
preds = Dense(len(macronum), activation='softmax')(l_lstm)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("Bidirectional LSTM")
model.summary()



cp=ModelCheckpoint('model_cnn.hdf5',monitor='val_acc',verbose=1,save_best_only=True)

history=model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=6, batch_size=200,callbacks=[cp])

y_predict = model.predict(x_val)


        

    
# Processing test data to make prediction
df_test = pd.read_csv('test.tsv', sep='\t', header=0)
df_test = df_test.dropna()
df_test = df_test.reset_index(drop = True)


print('Shape of dataset ',df_test.shape)
print(df_test.columns)
print('No. of unique classes',len(set(df_test['id'])))

texts_test = []
testlist = []
for idx in range(df_test.comment.shape[0]):
    
     text_test = BeautifulSoup(df_test.comment[idx])
     texts_test.append(str(text_test.get_text().encode()))


for idx in df_test['id']:
    testlist.append(idx)


print(df_test.comment.shape[0])





print(len(texts_test))
#tokenizer_test = Tokenizer(num_words=MAX_NB_WORDS)
#tokenizer_test.fit_on_texts(texts_test)
sequences_test = tokenizer.texts_to_sequences(texts_test)

#word_index_test = tokenizer.word_index

data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

#indices_test = np.arange(data_test.shape[0])
#np.random.shuffle(indices_test)
#data_test = data_test[indices_test]



#prediction
final_pre = model.predict(data_test)


finalpre_list = []
for x,y in final_pre:
    if x > y:
        finalpre_list.append(0);
    else:
       finalpre_list.append(1);
       

#Write into csv
with open('fileName.csv', 'w') as f:
    writer = csv.writer(f)
    count = 0
    writer.writerow(["id", "label"])
    for item in set(df_test['id']):
       writer.writerow([str(item), str(finalpre_list[count])])
       count += 1
       


    