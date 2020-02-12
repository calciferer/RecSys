import time


def curTime() -> str:
    """
    :return: 当前时间，以字符串表示
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

if __name__ == '__main__':
    print(curTime())
