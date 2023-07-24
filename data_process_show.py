import jieba
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def data_process():
    # todo:读取数据，重命名列名['Id','date','content','label']
    df = pd.read_excel('./data/dataProcess.xlsx', header=0).iloc[:, :].astype(str)
    df.columns = ['Id', 'date', 'content', 'label']
    # 对标签进行编码，Negative=0,Positive=1
    df['label'][df['label'] == 'Negative'] = 0
    df['label'][df['label'] == 'Positive'] = 1
    return df


def data_process_show():
    # 对标签Negative，Positive的占比进行可视化
    # 对评论标签的Negative == 0、Positive == 1的占比，用饼状图展示分布
    df = data_process()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    size = df.iloc[:, 3:].value_counts()
    labels = ['0', '1']
    colors = ['red', 'blue']
    plt.figure(figsize=(6, 6))
    explode = [0.01, 0.01]
    plt.pie(size, colors=colors, explode=explode, labels=labels, autopct='%1.1f%%')
    plt.title('占比')
    plt.show()

    # 按年份，对标签Negative、Positive的占比进行统计，用柱状图展示分布情况
    x = np.arange(8)
    bar_width = 0.35
    tick_label = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d %H:%M:%S')
    df['date'] = df['date'].dt.year
    P, N = [], []
    for i in range(2011, 2019):
        df_new = df.loc[df['date'] == i]
        df1 = df_new.loc[df_new['label'] == 0]
        df2 = df_new.loc[df_new['label'] == 1]
        N.append(len(df1))
        P.append(len(df2))
    plt.bar(x, N, bar_width, color='r', align='center', label='0', alpha=0.5)
    plt.bar(x + bar_width, P, bar_width, color='b', align='center', label='1', alpha=0.5)
    plt.xlabel('year')
    plt.ylabel('0/1')
    plt.xticks(x + bar_width / 2, tick_label)
    plt.legend()
    plt.show()

    # 对积极正面的评论内容，以词云图展示
    def seg(sentence):
        with open("./data/hit_stopwords.txt", "r", encoding="utf-8") as f:
            stopword = [one.strip() for one in f.readlines()]
        clear_list = []
        word_list = jieba.cut(sentence)
        for one in word_list:
            if one not in stopword:
                clear_list.append(one)
        sent = ' '.join(clear_list)
        return sent

    df['pre'] = df['content'].apply(seg)  # 特征工程
    vec = TfidfVectorizer(ngram_range=(2, 2))
    x = vec.fit_transform(df['pre'])
    k = KMeans(n_clusters=2, random_state=10).fit(x)
    df['result'] = k.labels_
    for i in df['result'].unique():
        w = WordCloud(width=1000, height=1000, font_path="data/msyhl.ttc")
        a = df.loc[df['result'] == i, 'pre']
        words = ' '.join(a)
        w.generate(words, )
        w.to_image()
        w.to_file('./picture/cluster_{}.png'.format(i))


if __name__ == '__main__':
    data_process_show()
