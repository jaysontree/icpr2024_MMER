[简体中文](README.md) | English

# icpr2024_MMER

3rd ranking approach for ICPR 2024 Multi-line Mathematical Expression Recognition Competition

main contributor: JinQiu [@kingqiuol](https://github.com/kingqiuol)

### Background and discussion

we have explored serveral formula recognition approaches 
- ABM is two directional attention aggregation model. (https://github.com/XH-B/ABM)
- CoMER is transformer-based model. (https://github.com/Green-Wood/CoMER)
- CAN adds a counting module to ABM-like model. and we achieved 4th ranking in ICDAR 2023 Recognition Multi-line Handwritten Mathematical Expression Competition using CAN, with additional large conv kernel for better converage of large math symbols. (https://github.com/LBH1024/CAN)
- ICAL proposed a implicit character module which adds context-aware information to help predict.(https://github.com/qingzhenduyu/ICAL)
- Nougat-OCR is fine-tunning approach from facebook/nougat (https://github.com/NormXU/nougat-latex-ocr)

| Model | HME100K ExpRate | Year |
| ----- | ------- | ---- |
| BTTR  | 64.1    | 2021 |
| CAN   | 67.31   | 2022 |
| CoMER | 68.12   | 2022 |
| LAST  | ——      | 2023 |
| ICAL  | 69.25   | 2024 |

### MMER Dataset

```
Total Num: 15000

Height:
- Mean: 416.7778
- Median: 374.0
- Max: 2483
- Min: 115
Weight：
- Mean: 1615.7956666666666
- Median: 1525.5
- Max: 4446
- Min: 181

Label Length(number of symbols):
- Mean: 145.4
- Median: 128
- Max: 1050
- Min: 13

Label Length distribution:
```
<img src="img\2.png" alt="2" style="zoom:72%;" />

```
Label Dict: 337
Label Dict distribution:
```
<img src="img\1.png" alt="1" style="zoom: 33%;" />

```
Training setup:
- Train:val = 9:1 # refer to script/preprocess.ipynb
```
### Implementation Results
|        | ExpRate | TokenAcc |
| ------ | ------- | -------- |
| CAN    | 0.34    | 0.79     |
| CoMER  | 0.62    | 0.95     |
| ICAL   |    -    |     -    |
| Nougat | 0.66    | 0.98     |
