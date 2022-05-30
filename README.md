# 基于朴素贝叶斯的垃圾邮件识别

## 目录结构

```
.
├── README.md
├── data
│   └── SMSSpamCollection.txt
├── main.py
├── models
│   ├── NaiveBayes.py
│   └── __init__.py
├── notebooks
│   └── spam.ipynb
├── requirements.txt
└── utils
    ├── __init__.py
    └── load_data.py
```

* `data/SMSSpamCollection.txt`：为垃圾邮件数据集
* `main.py`：程序入口，调用模块进行训练与检测（未添加准确率统计等功能，可自行实现）
* `models/NaiveBayes.py`：贝叶斯模型（基于sklearn实现，可改为自己的实现）
* `notebooks/spam.ipynb`：`jupyter`记事本
* `utils/load_data.py`：加载数据集方法
