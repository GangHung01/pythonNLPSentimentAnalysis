import jieba
import warnings
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
from tensorflow.python.keras.models import load_model
from AffectiveAnalysis.data_process_show import data_process

warnings.filterwarnings('ignore')


def Analysis_Sentence():
    df = data_process()
    df1 = df[['content', 'label']]

    cw = lambda x: list(jieba.cut(x))
    df1['content'] = df1['content'].apply(cw)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df1['content'])
    word2id = tokenizer.word_index

    x_train = tokenizer.texts_to_sequences(df1['content'])

    max_len = max(map(len, x_train))
    x_train = pad_sequences(x_train, maxlen=max_len)

    label2id = {i: id for id, i in enumerate(df1['label'].unique())}
    y_train = df1['label'].apply(lambda x: label2id[x])
    # y_test = test_data['label'].apply(lambda x: label2id[x])

    inputs = Input(shape=(None,))
    embedding = Embedding(len(word2id) + 1, 300)(inputs)  # 输入，词嵌入
    # 词窗大小3，4，5
    convs = []
    for kernel_size in [3, 4, 5]:
        c = Conv1D(128, kernel_size, activation='relu')(embedding)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    # 合并三个输出
    x = Concatenate()(convs)
    drop = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax')(drop)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(x_train, y_train, batch_size=64, epochs=10)
    model.save('./model/model.h5')


def sim_sen_Result(sentence):
    df = data_process()
    df1 = df[['content', 'label']]

    cw = lambda x: list(jieba.cut(x))
    df1['content'] = df1['content'].apply(cw)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df1['content'])
    # word2id = tokenizer.word_index
    x_train = tokenizer.texts_to_sequences(df1['content'])
    max_len = max(map(len, x_train))

    model = load_model('./model/model.h5')
    sentence = sentence
    sentence = jieba.cut(sentence)
    tokenizer = Tokenizer()
    x = tokenizer.texts_to_sequences(sentence)
    x_test = pad_sequences(x, maxlen=max_len)
    predict = model.predict(x_test)
    predict = np.argmax(predict, axis=1)
    print(predict)
