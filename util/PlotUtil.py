"""
绘图工具类
"""
import matplotlib.pyplot as plt


def show(xLable: str, yLable: str, xValues, yValues):
    plt.xlabel(xLable)
    plt.ylabel(yLable)
    plt.plot(xValues, yValues)
    plt.show()
