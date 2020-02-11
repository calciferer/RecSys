"""
评测指标
"""
import math

import pandas as pd


def precision(Ru: {str: set}, Tu: {str: set}) -> float:
    """
    准确率(TopN评测指标):【精确率是针对我们`预测结果`而言的，它表示的是预测为正的样本中有多少是真正的正样本，
    例如我们给用户推荐了100条新闻，其中10条用户产生了点击，那么准确率为10/100 = 0.1】

    公式为:对于每个u，求和(len(Ru & Tu))/求和(len(Ru))
    :param Ru: 全部user的推荐列表
    :param Tu: 测试集
    :return:准确率
    """
    hit = 0
    len_Ru = 0
    for u in Ru:
        hit += len(Ru[u] & Tu[u])
        len_Ru += len(Ru[u])
    return hit / len_Ru


def recall(Ru: {str: set}, Tu: {str: set}) -> float:
    """
    【召回率(TopN评测指标):召回率是针对我们原来的样本而言的，它表示的是样本中的正例有多少被预测正确了，
    例如我们给用户推荐了100条新闻，其中10条用户产生了点击，而用户最终在平台上总共点击了200条新闻，
    那么召回率为10 / 200 = 0.05，表示的是推荐系统推荐的那些符合用户兴趣并产生点击的新闻量占了用户实际总共点击的新闻 有多少比例 】

    公式为:对于每个u，求和(len(Ru & Tu))/求和(len(Tu))
    :param Ru: 全部user的推荐列表
    :param Tu: 测试集
    :return:召回率
    """
    hit = 0
    len_Tu = 0
    for u in Ru:
        hit += len(Ru[u] & Tu[u])
        len_Tu += len(Tu[u])
    return hit / len_Tu


def coverage(Ru: {str: set}, N_items: int) -> float:
    """
    覆盖率：表示给全部用户推荐出来的item，占全部item的比例,覆盖率反映了推荐算法发掘长尾的能力，
    覆盖率越高，说明推荐算法越能够将长尾中的物品推荐给用户
    :param Ru: 全部user的推荐列表
    :param N_items:训练集全部item个数
    :return:覆盖率
    """
    rec_items = set()
    for v in Ru.values():
        rec_items.update(v)
    return len(rec_items) / N_items * 1.0


def popularity(R_train: pd.DataFrame, Ru: {str: set}) -> float:
    """
    流行度：同时可以评测推荐的新颖度，如果推荐出的物品都很热门，说明推荐的新颖度较低，否则说明推荐结果比较新颖。
    单个item的流行度为该item在训练集中出现的次数。推荐结果的流行度为：推荐出来的每个物品的流行度求log平均值
    :param R_train:训练集评分矩阵
    :param Ru:推荐列表
    :return:流行度
    """
    # 计算每个item的流行度
    # 评分为1的话，直接每一列求和
    item_pop = R_train.sum()
    # 对Ru中的每个item
    pop = 0.0
    n = 0
    for items in Ru.values():
        for item in items:
            pop += math.log(1 + item_pop[item])
            n += 1
    return pop / n
