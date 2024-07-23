[简体中文](README.md) | English

# icpr2024_MMER

3rd ranking approach for ICPR 2024 Multi-line Mathematical Expression Recognition Competition

main contributor: JinQiu [@kingqiuol](https://github.com/kingqiuol)

### background and discussion

we have explored serveral formula recognition approaches 
- ABM is two directional attention aggregation model. (https://github.com/XH-B/ABM)
- CoMER is transformer-based model. (https://github.com/Green-Wood/CoMER)
- CAN adds a counting module to ABM-like model. and we achieved 4th ranking in ICDAR 2023 Recognition Multi-line Handwritten Mathematical Expression Competition using CAN, with additional large conv kernel for better converage of large math symbols. (https://github.com/LBH1024/CAN)
- ICAL proposed a implicit character module which adds context-aware information to help predict.(https://github.com/qingzhenduyu/ICAL)
- Nougat-OCR is fine-tunning approach from facebook/nougat (https://github.com/NormXU/nougat-latex-ocr)
