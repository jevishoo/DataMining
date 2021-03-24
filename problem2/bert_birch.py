"""
    @author: Jevis_Hoo
    @Date: 2020/11/29 7:40
    @Description: 
"""
from __future__ import absolute_import
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans
import jieba
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import torch
from bert.tokenization import BertTokenizer
from bert.modeling import BertModel
from utils.util import load_corpus, load_users, save_cluster, gen_ans, colors


def get_stop_words():
    stopwords = []
    f1 = open("./data/baidu_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("./data/cn_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("./data/hit_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("./data/scu_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    return stopwords


# 对句子进行分词
def seg_sentence(sentence, stop_words):
    sentence_seged = jieba.cut(sentence)
    outstr = ''
    for word in sentence_seged:
        if word not in stop_words:
            if word != '\t':
                outstr += word
    return outstr


def birch(X, k):  # 待聚类点阵,聚类个数
    clusterer = Birch(n_clusters=k)

    y = clusterer.fit_predict(X)

    return y


def Silhouette(X, y):
    from sklearn.metrics import silhouette_samples, silhouette_score

    silhouette_avg = silhouette_score(X, y)  # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(X, y)  # 每个点的轮廓系数

    print(silhouette_avg)

    return silhouette_avg, sample_silhouette_values


def main():
    # random_num = np.random.choice([i for i in range(10000)])
    random_num = 1
    data = load_corpus()

    "加载预训练模型的分词器、词汇表"
    # tokenizer = BertTokenizer.from_pretrained('/home/hezoujie/Models/roberta_pytorch')
    tokenizer = BertTokenizer.from_pretrained('/home/hezoujie/Models/roberta_pytorch')

    # '加载预训练模型'
    # model = BertModel.from_pretrained('/home/hezoujie/Models/roberta_pytorch')  # BERT中文模型的路径
    model = BertModel.from_pretrained('/home/hezoujie/Models/roberta_pytorch')  # BERT中文模型的路径
    # 模型下载地址https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz

    # '''模型设置成评估模式，去除dropout（随机停止更新部分参数）的模块
    # 因为在评估时，有可重复的结果是很重要的'''
    model.eval()
    stop_words = get_stop_words()

    vectors = []
    for d in data:
        note = d.strip()
        note = re.sub("\s", "", note)  # 去除空白字符
        note = seg_sentence(note, stop_words).replace('捆绑', '伊景园滨河苑捆绑')
        #     print(note)
        sent = "[CLS] " + note + " [SEP]"
        tokenized_text = tokenizer.tokenize(sent)
        #     print(tokenized_text)

        indexed_tokens = tokenizer.convert_tokens_to_ids(
            tokenized_text)  # Map the token strings to their vocabulary indeces.

        segments_ids = [1] * len(tokenized_text)

        #     print (segments_ids)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            # 当网络中的某一个tensor不需要梯度时，可以使用torch.no_grad()来处理。
            # See the models docstrings for the detail of the inputs
            try:
                outputs = model(tokens_tensor, segments_tensors)
                encoded_layers = outputs[0]  # 第一个元素，是bert模型的最后一层的隐藏状态。
                # print(encoded_layers.size())
                token_vecs = encoded_layers[0]  # 得到一句话里每个词的768维向量的矩阵。
                # print(token_vecs.size())
                sentence_embedding = torch.mean(token_vecs[0], dim=0)
                # print(sentence_embedding.size())
                note_vec = sentence_embedding.numpy().tolist()  # 留言的向量表示，list格式
                vectors.append(note_vec)
            except:
                print("提取向量有误")
                continue

    k = 300
    cluster = birch(vectors, k)
    print("Done")

    silhouette_avg, sample_silhouette_values = Silhouette(vectors, cluster)  # 轮廓系数
    print("轮廓：{} {}".format(silhouette_avg, sample_silhouette_values))
    silhouette_values = pd.DataFrame(sample_silhouette_values, columns=["value"])
    silhouette_values.to_csv("value_{}.csv".format(random_num), index=None)

    res = save_cluster(cluster, data, k, random_num)
    gen_ans(res, cluster)

    news = []
    label_new = []
    for ind, i in enumerate(cluster):
        if i in res:
            news.append(vectors[ind])
            label_new.append(i)
    X = np.array(news)
    labels = label_new

    svd = TruncatedSVD(n_components=2).fit(X)
    datapoint = svd.transform(X)

    plt.figure(figsize=(8, 5))
    label1 = list(colors.values())

    color = [label1[i // 4] for i in labels]
    sns.scatterplot(datapoint[:, 0], datapoint[:, 1], hue=color)
    plt.savefig('cluster{}.pdf'.format(random_num), dpi=500)

    pca = PCA(n_components=3)  # 初始化PCA
    datapoint = pca.fit_transform(X)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(datapoint[:, 0], datapoint[:, 1], hue=color)
    plt.savefig('cluster_pca{}.pdf'.format(random_num), dpi=500)

    set_labels = set(labels)
    data_point = {}

    for label in set_labels:
        data_point[label] = []

    for i, label in enumerate(labels):
        data_point[label].append(datapoint[i].tolist())

    ax = plt.subplot(111, projection='3d')
    for key in data_point.keys():
        ax.scatter(np.array(data_point[key])[:, 0], np.array(data_point[key])[:, 1],
                   np.array(data_point[key])[:, 2])  # 绘制数据点

    plt.savefig('birch_cluster_pca_3d.pdf', dpi=500)

    kmeans = KMeans(n_clusters=k)
    k_cluster = kmeans.fit_predict(vectors)

    news = []
    label_new = []
    for ind, i in enumerate(k_cluster):
        if i in res:
            news.append(vectors[ind])
            label_new.append(i)
    X = np.array(news)
    labels = label_new

    print(labels)

    svd = TruncatedSVD(n_components=3).fit(X)
    datapoint = svd.transform(X)

    plt.figure(figsize=(8, 5))
    label1 = list(colors.values())

    color = [label1[i // 4] for i in labels]
    sns.scatterplot(datapoint[:, 0], datapoint[:, 1], hue=color)
    plt.savefig('k_cluster_svd_{}.pdf'.format(random_num), dpi=500)

    pca = PCA(n_components=3)  # 初始化PCA
    datapoint = pca.fit_transform(X)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(datapoint[:, 0], datapoint[:, 1], hue=color)
    plt.savefig('k_cluster{}.pdf'.format(random_num), dpi=500)

    set_labels = set(labels)
    data_point = {}

    for label in set_labels:
        data_point[label] = []

    for i, label in enumerate(labels):
        data_point[label].append(datapoint[i].tolist())

    ax = plt.subplot(111, projection='3d')
    for key in data_point.keys():
        ax.scatter(np.array(data_point[key])[:, 0], np.array(data_point[key])[:, 1],
                   np.array(data_point[key])[:, 2])  # 绘制数据点
    plt.savefig('kmeans_cluster_pca_3d.pdf', dpi=500)


if __name__ == '__main__':
    main()
