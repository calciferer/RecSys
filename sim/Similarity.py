"""
相似度计算
"""
import math
import numpy as np
from numpy.linalg import norm


def cosineSim(a, b, dtype=np.ndarray):
    """
    余弦相似度。如果是集合,公式为：a与b的交集的长度/根号(a的长度*b的长度);
    如果是向量，公式为a与b的点积/a的范数*b的范数
    :param dtype: set或np.ndarray
    :param a: 集合或向量a
    :param b: 集合或向量b
    :return: a与b的余弦相似度
    """
    if dtype == set:
        return len(a & b) / math.sqrt(len(a) * len(b))
    elif dtype == np.ndarray:
        return np.dot(a, b) / (norm(a) * norm(b))
    else:
        return None


def jaccardSim(a: set, b: set) -> float:
    """
    Jaccard相似度，公式为：a与b的交集的长度/a与b的并集的长度
    :param a:
    :param b:
    :return: a与b的Jaccard相似度
    """
    return len(a & b) / len(a | b)


if __name__ == '__main__':
    a = np.array([1, 0, 0])
    b = np.array([1, 1, 0])
    print(cosineSim(a,b))
