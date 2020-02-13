# RecSys

推荐系统算法总结

本项目实现了以下推荐算法

|  算法    | 解析 | 文件 | 算法评估 |
| :--: | :--: | :--: | :--: |
| UserCF | [【推荐系统】算法总结(1) UserCF](https://blog.csdn.net/weixin_42818402/article/details/104271902) |  [UserCF.py](https://github.com/calciferer/RecSys/blob/master/algo/UserCF.py)| [result](https://github.com/calciferer/RecSys/tree/master/result/UserCF) |
| ItemCF | [【推荐系统】算法总结(2) ItemCF](https://blog.csdn.net/weixin_42818402/article/details/104290503) | [ItemCF](https://github.com/calciferer/RecSys/blob/master/algo/ItemCF.py) | [result](https://github.com/calciferer/RecSys/tree/master/result/ItemCF) |

# 运行

```python
# 直接运行
if __name__ == '__main__':
    userCF = UserCF() 
    userCF.K = 20 # 可以设置相关参数和变量，在UserCF模块属性中
    userCF.run() 
```

```python
# 分步调试

if __name__ == '__main__':
    userCF = UserCF()
    userCF.load_data() # 加载数据
    userCF.calc_user_sim() # 计算用户相似度
    result = pd.DataFrame(columns=['K', 'N', "precision", 'recall', 'cov', 'pop']) # 评估准确率，召回率，覆盖率，流行度
    for index, K in enumerate(range(5, 41)):
        userCF.K = K
        userCF.rec()
        precision, recall, cov, pop = userCF.evaluate()
        result.loc[index] = K, userCF.N, precision, recall, cov, pop
				SaveHelper.save(result, 'UserCF')
```

