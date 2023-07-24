import jieba
import numpy as np
import pandas as pd
from gensim import models, corpora, similarities
from AffectiveAnalysis.data_process_show import data_process


def Similar_Sentence():
    #df = pd.read_excel('./data/dataProcess.xlsx', header=0).iloc[:, :].astype(str)
    #df.columns = ['Id', 'date', 'content', 'label']
    df = data_process()
    org_text = df['content'].values
    # 首先，对评论分词，并去掉停用词。
    with open("./data/hit_stopwords.txt", "r", encoding="utf-8") as f:
        stopword = [one.strip() for one in f.readlines()]
    documents = [jieba.lcut(doc) for doc in org_text]
    sentences = [[word for word in doc if word not in stopword] for doc in documents]
    # 构建词典，给每个词编号
    dictionary = corpora.Dictionary(sentences)
    # 每条评论里每个词的出现频次
    corpus = [dictionary.doc2bow(text) for text in sentences]

    lsi = models.LsiModel(corpus, id2word=dictionary, power_iters=100, num_topics=10)
    # lsi[corpus] 是所有评论对应的向量
    index = similarities.MatrixSimilarity(lsi[corpus])

    word = '蚊帐放下后落不到地板，容易进蚊子，'
    # 词袋模型，统计词频
    vec_bow = dictionary.doc2bow(jieba.lcut(word))
    # 计算 query 对应的向量
    vec_lsi = lsi[vec_bow]
    # 计算每条评论与word的相似度
    sims = index[vec_lsi]
    # 输出（原始文档，相似度）二元组
    result = [(org_text[i[0]], i[1]) for i in enumerate(sims)]
    # 按照相似度逆序排序,
    result_sort = sorted(result, key=lambda x: -x[1])
    # 去除第一句，第一句和word相同的一句话
    new_result = result_sort[1:6]

    user = []
    for i in range(len(new_result)):
        user.append(df.loc[df['content'] == new_result[i][0], ['Id']])
    user = np.ravel(user)

    print(pd.DataFrame([list(t) for t in zip(user, new_result)], columns=['用户', '相似评论和相似评论的概率']))

