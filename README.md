# icpr2024_MMER

Repo for Multi-line Math Expression Recognitions Competition

## 一、数据分析

目前训练集总数为15000张图片，按照9：1划分数据集，详情见`script/preprocess.ipynb`

统计训练集相关详情如下：

* 图片分析：

```bash
===================>H的参数：
均值: 416.7778
中位数: 374.0
最大值: 2483
最小值: 115
===================>W的参数：
均值: 1615.7956666666666
中位数: 1525.5
最大值: 4446
最小值: 181
```

* 标签分析：

```bash
===================>标签长度参数：
均值: 145.39526666666666
中位数: 128.0
最大值: 1050
最小值: 13
```

目前标签长度分布如下：

<img src="img\2.png" alt="2" style="zoom:72%;" />

目前统计字典总数为：337，分布如下：

<img src="img\1.png" alt="1" style="zoom: 33%;" />

可以看出字典分布不均衡，长度最多为100长度左右。目前设置设置图片输入大小为$560\times560$​，最大长度为1024。

## 二、模型选择

目前主流的公式识别模型架构为encoder-decoder，从2021到2024发展的模型如下：

| 模型  | HME100K | 时间 |
| ----- | ------- | ---- |
| BTTR  | 64.1    | 2021 |
| CAN   | 67.31   | 2022 |
| CoMER | 68.12   | 2022 |
| LAST  | ——      | 2023 |
| ICAL  | 69.25   | 2024 |

当然也有Nougat(2023)做过相关工作。

目前主要测试了**CAN**、**CoMER**、**ICAL**、**Nougat**，等模型，测试baseline结果如下：

|        | ExpRate | TokenAcc |
| ------ | ------- | -------- |
| CAN    | 0.34    | 0.79     |
| CoMER  |         |          |
| ICAL   |         |          |
| Nougat | 0.66    | 0.98     |

