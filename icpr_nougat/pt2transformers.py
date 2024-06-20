# -*- coding:utf-8 -*-
# create: @time: 10/8/23 11:47
import argparse
import os
import tqdm
import cv2
import shutil


import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from dataset.processor import NougatLaTexProcessor
from dataset.donut_dataset import decode_text


def parse_option():
    parser = argparse.ArgumentParser(prog="nougat inference config", description="model archiver")
    parser.add_argument("--pretrained_model_name_or_path", default="./cust-data/weights")
    parser.add_argument("--cust_data_init_weights_path", default="./nougat-small")
    parser.add_argument("--device", default="cpu")
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
    state = torch.load('./nougat_latex/nougat-small_epoch15_step7136_lr1.952076e-06_avg_loss0.05098_token_acc0.78397_edit_dis0.33855.pth', map_location=args.device)
    model.load_state_dict(state)
    model.save_pretrained(args.cust_data_init_weights_path)
    
    # init processor
    tokenizer = NougatTokenizerFast.from_pretrained(args.pretrained_model_name_or_path)
    latex_processor = NougatLaTexProcessor.from_pretrained(args.pretrained_model_name_or_path)
    vocab = tokenizer.get_vocab()
    # print(vocab)
    latex_processor.save_pretrained(args.cust_data_init_weights_path)
    
    for file in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'special_tokens_map.json']:
        shutil.copy(os.path.join(args.pretrained_model_name_or_path, file),
                    os.path.join(args.cust_data_init_weights_path, file))



if __name__ == '__main__':
    run_nougat_latex()
