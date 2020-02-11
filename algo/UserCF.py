import logging

import pandas as pd
from sklearn.metrics import pairwise_distances

from Splitter.Splitter import ratioSplitter
from evaluate import Evaluate
from util import FileUtil
import numpy as np
from util.Logger import logger
from util.PlotUtil import show


class UserCF:
    DataPath = "../dataset/ml-latest-small/ratings.csv"  # 全部数据路径
    Ratio = 0.8  # 训练集占数据集的比例
    K = 10  # 最邻近的K个user
    N = 10  # 为每个user推荐N个item
    R_train = None  # type:pd.DataFrame # 训练集评分矩阵
    R_test = None  # type:pd.DataFrame # 测试集评分矩阵
    users = None  # type:np.ndarray # 全部user列表
    items = None  # type:np.ndarray # 全部item列表
    W = None  # type:pd.DataFrame # user相似度矩阵
    P = None  # type:pd.DataFrame # 预测评分矩阵

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
        logger.debug(f"users长度:{len(users)}")
        logger.debug(f"items长度:{len(items)}")

        # 构建训练集评分矩阵
        R_train = pd.DataFrame(np.zeros((len(users), len(items))), index=users, columns=items)
        for r in train:
            R_train.at[r.uid, r.iid] = 1  # 隐式反馈，这里就设置为1，忽略具体的评分数据
        logger.debug(f"训练集评分矩阵:\n{R_train}")

        # 构建测试集评分矩阵
        test_users = np.unique([r.uid for r in test])
        test_items = np.unique([r.iid for r in test])
        R_test = pd.DataFrame(np.zeros((len(test_users), len(test_items))), index=test_users, columns=test_items)
        for r in test:
            R_test.at[r.uid, r.iid] = 1  # 隐式反馈，这里就设置为1，忽略具体的评分数据
        logger.debug(f"测试集评分矩阵:\n{R_test}")

        self.users = users
        self.items = items
        self.R_train = R_train
        self.R_test = R_test

        return users, items, R_train, R_test

    def calc_user_sim(self):
        """
        2. 计算user相似度。直接使用sklearn的pairwise_distances函数来计算，速度大大提高
        """
        W = 1 - pairwise_distances(self.R_train.to_numpy(), metric="cosine")
        W = pd.DataFrame(W, index=self.users, columns=self.users)
        logger.debug(f"user相似度矩阵：\n{W}")
        self.W = W
        return W

    def rec(self):
        """
        3. 推荐。计算推荐(兴趣)矩阵P，P[u][i]表示u对i的兴趣值(预测的u对i的评分)
        """
        logger.info(f"开始推荐,K={self.K}")
        P = pd.DataFrame(np.zeros((len(self.users), len(self.items))), index=self.users, columns=self.items)
        for u in self.users:
            K_Wu = self.W[u].nlargest(self.K + 1).iloc[1:]
            for v, wuv in K_Wu.items():
                Rv = self.R_train.loc[v]
                vis = Rv[Rv != 0.0].index.values
                for i in vis:
                    # 如果i已经在u的评分列表中，则跳过
                    if self.R_train.at[u, i] != 0.0:
                        continue
                    P.at[u, i] += self.W.at[u, v] * self.R_train.at[v, i]

        logger.debug(f"预测矩阵：\n{P}")
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

        logger.info(
            f"K={self.K},N={self.N},准确率{precs * 100:.2f}%"
            f",召回率{recl * 100:.2f}%,覆盖率{covrg * 100:.2f}%,流行度{popu}")
        return precs, recl, covrg, popu

    def run(self):
        """
        执行上面4个步骤,也可以在外部调用，分步调试
        """
        self.load_data()
        self.calc_user_sim()
        self.rec()
        self.evaluate()


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    userCF = UserCF()
    userCF.load_data()
    userCF.calc_user_sim()

    Ks = np.arange(5, 41)
    precisions = []
    recalls = []
    covs = []
    pops = []

    for K in Ks:
        userCF.K = K
        userCF.rec()
        precision, recall, cov, pop = userCF.evaluate()
        precisions.append(precision)
        recalls.append(recall)
        covs.append(cov)
        pops.append(pop)

    show('K', "precision", Ks, np.array(precisions))
    show('K', "recall", Ks, np.array(recalls))
    show('K', "coverage", Ks, np.array(covs))
    show('K', "popularity", Ks, np.array(pops))
