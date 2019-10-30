##基于keras和sklearn的文本分类任务
###数据来源
CCL2019“小牛杯”中文幽默计算

任务说明参见CCL2019官方网站：[http://www.cips-cl.org/static/CCL2019/call-evaluation.html ](http://www.cips-cl.org/static/CCL2019/call-evaluation.html )

官方数据及baseline:[https://github.com/DUTIR-Emotion-Group/CCL2019-Chinese-Humor-Computation](https://github.com/DUTIR-Emotion-Group/CCL2019-Chinese-Humor-Computation)

###程序架构
    |-- data
    	|-- train.csv        #训练集数据
        |-- development.csv  #测试集数据
    |-- models
        |--Machine_Learning
        |--Deep_Learning
    |-- main.py              # 模型训练和测试
    |-- preprocessing.py     # 预处理

###使用说明
在main函数中选择需要的模型，运行以下python语句：

    python main.py



