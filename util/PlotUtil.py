"""
绘图工具类
"""
import matplotlib.pyplot as plt
import pandas as pd


def showAndSave(result: pd.DataFrame, dir: str):
    """
    显示并保存图像,数据
    """
    imgNames = ["precision", 'recall', 'cov', 'pop']
    for imgName in imgNames:
        plt.xlabel("K")
        plt.ylabel(imgName)
        plt.plot(result['K'].values, result[imgName].values)
        plt.savefig(dir + '/' + imgName + '.png')
        plt.show()


if __name__ == '__main__':
    result = pd.DataFrame(columns=['K', 'N', "precision", 'recall', 'cov', 'pop'])
    result.loc[0] = 10, 20, 0.5, 0.4, 0.8, 5
    result.loc[1] = 20, 20, 0.6, 0.6, 0.6, 6
    result.to_html("../result/ItemCF/result.html")

    showAndSave(result, '../result/ItemCF')
