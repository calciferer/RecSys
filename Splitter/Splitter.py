"""
划分训练集,测试集
"""
from sklearn.utils import shuffle


def ratioSplitter(data: list, ratio: float, *, fixed: bool = False) -> tuple:
    """
    按比例划分
    :param fixed: 是否每次结果一致，加入random_state参数表示每次shuffle结果一致
    :param data: 数据集
    :param ratio: 训练集占全部数据的比例
    :return: train,test
    """
    if fixed:
        data = shuffle(data, random_state=0)
    else:
        data = shuffle(data)
    pivot = int(len(data) * ratio)
    return data[:pivot], data[pivot:]


if __name__ == '__main__':
    print(ratioSplitter([1, 2, 3, 4, 5, 6], 0.5))
