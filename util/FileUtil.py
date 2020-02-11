"""
读写文件
"""

from Entities import Rating


def readRatings(*, fileName: str) -> [Rating]:
    """
    读取评分文件
    :param fileName: 文件名
    :return: ratings数组,格式为[namedtuple(int,int,float)]
    """
    ratings = []

    with open(fileName, 'r') as f:
        next(f)
        for line in f:
            s = line.split(',')
            uid = int(s[0])
            iid = int(s[1])
            rating = float(s[2])
            r = Rating(uid, iid, rating)
            ratings.append(r)
    return ratings


if __name__ == '__main__':
    data = readRatings(fileName="../dataset/ml-latest-small/ratings.csv")
    print(data)
    print(len(data))
