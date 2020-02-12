# RecSys

推荐系统算法总结

本项目实现了以下推荐算法

|  算法    | 解析 | 文件 | 算法评估 |
| :--: | :--: | :--: | :--: |
| UserCF | [【推荐系统】算法总结(1) UserCF](https://blog.csdn.net/weixin_42818402/article/details/104271902) |  [UserCF.py](https://github.com/calciferer/RecSys/blob/master/algo/UserCF.py)| [result](https://github.com/calciferer/RecSys/tree/master/result/UserCF) |

# 运行

```python
# 直接运行
if __name__ == '__main__':
  	logger.setLevel(logging.INFO) # DEBUG会输出调试信息，INFO会输出简要信息
    userCF = UserCF() 
    userCF.K = 20 # 可以设置相关参数和变量，在UserCF模块属性中
    userCF.run() 
```

```python
# 分步调试
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    userCF = UserCF()
    userCF.load_data() # 加载数据
    userCF.calc_user_sim() # 计算用户相似度
    Ks = np.arange(5, 41)
    for K in Ks:
        userCF.K = K
        userCF.rec() # 推荐
       	userCF.evaluate() #评估
```

