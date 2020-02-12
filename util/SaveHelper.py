# 保存运行结果
import pandas as pd
import os
from util import TimeUtil, PlotUtil


def save(result: pd.DataFrame, algo: str):
    """
    每次结果都新建一个文件夹来保存
    """
    basePath = "../result/" + algo
    if not os.path.exists(basePath):
        os.mkdir(basePath)
    ct = TimeUtil.curTime()
    os.mkdir(basePath + '/' + ct)
    result.to_html(basePath + '/' + ct + "/result.html")
    PlotUtil.showAndSave(result, basePath+'/'+ct)


if __name__ == '__main__':
    result = pd.DataFrame(columns=['K', 'N', "precision", 'recall', 'cov', 'pop'])
    result.loc[0] = 10, 20, 0.5, 0.4, 0.8, 5
    result.loc[1] = 20, 20, 0.6, 0.6, 0.6, 6
    save(result, 'ItemCF')
