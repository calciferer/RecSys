import logging

import pandas as pd
from sklearn.metrics import pairwise_distances

from Splitter.Splitter import ratioSplitter
from evaluate import Evaluate
from util import FileUtil, SaveHelper, Logger
import numpy as np


class ItemCF:
    DataPath = "../dataset/ml-latest-small/ratings.csv"  # 全部数据路径
    Ratio = 0.8  # 训练集占数据集的比例
    K = 10  # 最邻近的K个item
    N = 10  # 为每个user推荐N个item
    R_train = None  # type:pd.DataFrame # 训练集评分矩阵
    R_test = None  # type:pd.DataFrame # 测试集评分矩阵
    users = None  # type:np.ndarray # 全部user列表
    items = None  # type:np.ndarray # 全部item列表
    W = None  # type:pd.DataFrame # user相似度矩阵
    P = None  # type:pd.DataFrame # 预测评分矩阵

    logger = Logger.fileAndConsoleLogger('../result/ItemCF/itemCF.log')

    def load_data(self):
        """
        1. 读取原始数据。ratings数据格式为[Rating]
        """
        ratings = FileUtil.readRatings(fileName=self.DataPath)

        # 划分训练集和测试集
        train, test = ratioSplitter(ratings, self.Ratio, fixed=True)

        # 获取user和item列表,这里需要去重，并保持顺序
        users = np.unique([r.uid for r in train])
        items = np.unique([r.iid for r in train])
        self.logger.debug(f"users长度:{len(users)}")
        self.logger.debug(f"items长度:{len(items)}")

        # 构建训练集评分矩阵
        R_train = pd.DataFrame(np.zeros((len(users), len(items))), index=users, columns=items)
        for r in train:
            R_train.at[r.uid, r.iid] = 1  # 隐式反馈，这里就设置为1，忽略具体的评分数据
        self.logger.debug(f"训练集评分矩阵:\n{R_train}")

        # 构建测试集评分矩阵
        test_users = np.unique([r.uid for r in test])
        test_items = np.unique([r.iid for r in test])
        R_test = pd.DataFrame(np.zeros((len(test_users), len(test_items))), index=test_users, columns=test_items)
        for r in test:
            R_test.at[r.uid, r.iid] = 1  # 隐式反馈，这里就设置为1，忽略具体的评分数据
        self.logger.debug(f"测试集评分矩阵:\n{R_test}")

        self.users = users
        self.items = items
        self.R_train = R_train
        self.R_test = R_test

        return users, items, R_train, R_test

    def calc_item_sim(self):
        """
        2. 计算item相似度。直接使用sklearn的pairwise_distances函数来计算，速度大大提高
        """
        W = 1 - pairwise_distances(self.R_train.T.to_numpy(), metric="cosine")
        W = pd.DataFrame(W.T, index=self.items, columns=self.items)
        self.logger.debug(f"item相似度矩阵：\n{W}")
        self.W = W
        return W

    def rec(self):
        """
        3. 推荐。计算推荐(兴趣)矩阵P，P[u][i]表示u对i的兴趣值(预测的u对i的评分)
        """
        self.logger.info(f"开始推荐,K={self.K}")
        P = pd.DataFrame(np.zeros((len(self.users), len(self.items))), index=self.users, columns=self.items)
        for u in self.users:
            Ru = self.R_train.loc[u]
            uis = Ru[Ru != 0.0].index.values  # u有交互的items
            for i in uis:
                K_Wi = self.W[i].nlargest(self.K + 1).iloc[1:]  # 与i最近的K个item
                for j, wij in K_Wi.items():
                    # 如果j已经在u的评分列表中，则跳过
                    if self.R_train.at[u, j] != 0.0:
                        continue
                    P.at[u, j] += self.W.at[i, j] * self.R_train.at[u, i]

        self.logger.debug(f"预测矩阵：\n{P}")
        self.P = P
        return P

    def evaluate(self):
        """
        4. 评估效果
        """
        Ru_Dict = {}
        Tu_Dict = {}
        for u in self.users:
            # 为u推荐的N个item
            N_Pu = self.P.loc[u].nlargest(self.N).index.values
            # 测试集中u的items
            try:  # 测试集中可能没有训练集中全部的user
                Tu = self.R_test.loc[u]
            except KeyError:
                continue
            N_Tu = Tu[Tu != 0.0].index.values
            Ru_Dict[u] = set(N_Pu)
            Tu_Dict[u] = set(N_Tu)

        precs = Evaluate.precision(Ru_Dict, Tu_Dict)
        recl = Evaluate.recall(Ru_Dict, Tu_Dict)
        covrg = Evaluate.coverage(Ru_Dict, len(self.items))
        popu = Evaluate.popularity(self.R_train, Ru_Dict)

        self.logger.info(
            f"K={self.K},N={self.N},准确率{precs * 100:.2f}%"
            f",召回率{recl * 100:.2f}%,覆盖率{covrg * 100:.2f}%,流行度{popu}")
        return precs, recl, covrg, popu

    def run(self):
        """
        执行上面4个步骤,也可以在外部调用，分步调试
        """
        self.load_data()
        self.calc_item_sim()
        self.rec()
        self.evaluate()


if __name__ == '__main__':
    itemCF = ItemCF()
    itemCF.load_data()
    itemCF.calc_item_sim()

    result = pd.DataFrame(columns=['K', 'N', "precision", 'recall', 'cov', 'pop'])

    for index, K in enumerate(range(5, 41)):
        itemCF.K = K
        itemCF.rec()
        precision, recall, cov, pop = itemCF.evaluate()
        result.loc[index] = K, itemCF.N, precision, recall, cov, pop

    SaveHelper.save(result, 'ItemCF')
