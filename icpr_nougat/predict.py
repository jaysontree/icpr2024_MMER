# -*- coding:utf-8 -*-
# create: @time: 10/8/23 11:47
import argparse
import os
import tqdm
import cv2
import Levenshtein

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
# from utils.utils import process_raw_latex_code
from dataset.processor import NougatLaTexProcessor
from dataset.donut_dataset import decode_text

def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance

def evaluate_measure(str_algorithm, str_ground_truth): 
    # 编辑距离 insert + delete + replace 
    edit_dist = Levenshtein.distance(str_algorithm, str_ground_truth) 
    sum_len_two_str = len(str_algorithm) + len(str_ground_truth) 
    ratio = Levenshtein.ratio(str_algorithm, str_ground_truth)
    ldist = sum_len_two_str - (float(ratio) * float(sum_len_two_str)) 
    # 替换操作 
    replace_dist = ldist - edit_dist 
    if len(str_algorithm) > len(str_ground_truth): 
        more_word_error = len(str_algorithm) - len(str_ground_truth) 
        less_word_error = 0 
    else: 
        more_word_error = 0 
        less_word_error = len(str_ground_truth) - len(str_algorithm) 
    # - 平均识别率：[1 - (编辑距离 / max(1, groundtruth字符数, predict字符数))] * 100.0 % 的平均值； 
    recg_rate = 1 - (edit_dist / max(1, len(str_algorithm), len(str_ground_truth)))
    # print("识别率, 编辑距离, 替换错误, 漏字错误, 多字错误") 
    # print(recg_rate, edit_dist, replace_dist, less_word_error, more_word_error) 
    return recg_rate

def parse_option():
    parser = argparse.ArgumentParser(prog="nougat inference config", description="model archiver")
    parser.add_argument("--pretrained_model_name_or_path", default="./nougat-small")
    parser.add_argument("--cust_data_init_weights_path", default="./nougat-small")
    parser.add_argument("--device", default="gpu")
    return parser.parse_args()

def run_nougat_latex():
    args = parse_option()
    # device
    if args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # init model
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model_name_or_path).to(device)
    
    # init processor
    tokenizer = NougatTokenizerFast.from_pretrained(args.pretrained_model_name_or_path)
    latex_processor = NougatLaTexProcessor.from_pretrained(args.pretrained_model_name_or_path)
    vocab = tokenizer.get_vocab()

    # run test
    # image = Image.open('/home/qiujin/data/icpr_data/Test_A set/png/0072982.png')
    # if not image.mode == "RGB":
    #     image = image.convert('RGB')

    # pixel_values = latex_processor(image, return_tensors="pt").pixel_values
    # task_prompt = tokenizer.bos_token
    # decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False,
    #                               return_tensors="pt").input_ids
    # with torch.no_grad():
    #     outputs = model.generate(
    #         pixel_values.to(device),
    #         decoder_input_ids=decoder_input_ids.to(device),
    #         max_length=model.decoder.config.max_length,
    #         early_stopping=True,
    #         pad_token_id=tokenizer.pad_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #         use_cache=True,
    #         num_beams=10,
    #         bad_words_ids=[[tokenizer.unk_token_id]],
    #         return_dict_in_generate=True,
    #     )
        
    # print(outputs.sequences)
    # print(decode_text(outputs.sequences[0], vocab))
    
    
    with open("/hy-tmp/data/icpr_data/val.txt",encoding="utf-8") as f:
        lines = f.readlines()
    image_path = "/hy-tmp/data//icpr_data/"
    sum = 0
    line_right = 0
    total_time = 0
    e1, e2, e3 = 0, 0, 0
    recg_rate = 0
    for line in tqdm.tqdm(lines):
        name, *labels = line.split()
        input_labels = labels
        labels = ' '.join(labels)
        
        if not os.path.exists(os.path.join(image_path, name)):
            continue
        
        image = Image.open(os.path.join(image_path, name))
        
        if not image.mode == "RGB":
            image = image.convert('RGB')
            

        pixel_values = latex_processor(image, return_tensors="pt").pixel_values
        task_prompt = tokenizer.bos_token
        decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False,
                                    return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.decoder.config.max_length,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            
        prediction_src = decode_text(outputs.sequences[0], vocab)
        
        # def clear(latex):
        #     latex = latex.replace("\\geq","\\ge")
        #     latex = latex.replace("\\xlongequal { }","=")
        #     latex = latex.replace("\\xrightarrow { }","\\rightarrow")
        #     latex = latex.replace('\\Delta','\\triangle')
        #     latex = latex.replace('\\overline','\\bar')
        #     latex = latex.replace("\\cdots","\\cdot \\cdot \\cdot")
        #     latex = latex.replace('{','')
        #     latex = latex.replace('}','')
        #     latex = latex.replace(' ','')
        #     latex = latex.replace("^'","'")
        #     return latex
        sum+=1
        prediction = prediction_src
        # labels = clear(labels)
        
        recg_rate += evaluate_measure(prediction, labels)

        if prediction == labels:
            line_right += 1
        else:
            print("=======================", sum)
            print(name)
            print(prediction_src)
            print(prediction)
            print(labels)
        
        

        distance = compute_edit_distance(prediction, labels)   
        if distance <= 1:
            e1 += 1
        if distance <= 2:
            e2 += 1
        if distance <= 3:
            e3 += 1
    
    print(sum,line_right)
    print("平均识别率：", recg_rate/sum)
    print(f'ExpRate: {line_right / sum}')
    print(f'e1: {e1 / sum}')
    print(f'e2: {e2 / sum}')
    print(f'e3: {e3 / sum}')
    print("avg time:",total_time/sum)


if __name__ == '__main__':
    run_nougat_latex()
