import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

def gen_counting_label(labels, channel, tag):
    b, t = labels.size()
    device = labels.device
    counting_labels = torch.zeros((b, channel))
    if tag:
        ignore = [0, 1, 198, 199, 226, 228]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    return counting_labels.to(device)

# def gen_counting_label(labels, channel, tag,device=None):
#     b = len(labels)
#     # device = labels.device
#     counting_labels = torch.zeros((b, channel))
#     if tag:
#         # ignore = [0, 1, 107, 108, 109, 110]
#         # ignore = [0, 1, 192, 193, 220, 222]
#         ignore = [0, 1, 176, 177, 204, 206]
#     else:
#         ignore = []
#     for i in range(b):
#         for j in range(len(labels[i])):
#             k = labels[i][j]
#             if k in ignore:
#                 continue
#             else:
#                 counting_labels[i][k] += 1
#     return counting_labels.to(device)

def gen_ctc_label(labels):
    ctorch_labels = []
    ctorch_length = []
    b, t = labels.size()
    device = labels.device
    for i in range(b):
        labs = []
        for j in range(t):
            if labels[i][j]==0:
                break
            labs.append(labels[i][j])
            ctorch_labels.append(labels[i][j])
        ctorch_length.append(len(labs))
        
    return torch.LongTensor(ctorch_labels).to(device),torch.IntTensor(ctorch_length).to(device)


def get_features(batch_num, char_pos_return, embedding, label_return):
    """
    根据字符的位置从相应时间步中获取 features
    Args:
        char_pos_return: [[24, 42], [24, 46], [24, 49], [24, 52], [24, 53], [24, 55], [24, 60], [24, 62], [24, 66], [24, 69], [24, 74], [24, 78], [24, 117], [33, 52], [33, 55], [33, 57],[33, 60], [33, 63], [33, 117], [35, 38], [35, 43], ] 其中24表示batch中第24张图，42 指的是其在第42张图的第42个分片
        embedding: 输入全连接层的 tensor ----->   batch_size, w, c
    Returns:
        features: 字符对应的 feature
    """
    device = label_return.device
    features = torch.tensor([]).to(device)
    #embedding = embedding.cpu().detach().numpy() # (506, 119, 512)
    img_start_index = 0
    for key,value in batch_num.items():
        #print(img_start_index)
        feature_img = torch.tensor([]).to(device)
        
        # 这里key指的是符合条件（pred num == GT num）的batchsize中可用的图片id，因此下面去embedding的时候会有问题
        # print(key) # key 值应该是24,33
        img_num_of_batch = int(key) 
        for i in range(value):
            pos_index = char_pos_return[i+img_start_index][1].type(torch.long).to(device)
            feature_one_char = embedding[img_num_of_batch][pos_index,:].to(device)
            #print(len(feature_one_char)) # 512
            feature_img = torch.cat((feature_img, feature_one_char),0).to(device)
            #print(feature_img.size())
        features = torch.cat((features,feature_img),0).to(device)
        img_start_index = img_start_index + value
    features = features.view(-1, 512).to(device)
    #print(features.size())  # [6414, 512]
    new_label_return = np.array(label_return)
    label_return = torch.from_numpy(new_label_return).to(device)
    #print(label_return.size()) 
    #return None,None
    return features, label_return

def raw_pred_to_features(pred_label, length, labels, feature_embedding,num_classes):
    """得到用于计算 centerloss 的 embedding features，和对应的标签

    Args:
        pred_label (_type_):原始的预测结果，形如 [[6941, 6941, 0, 6941, 6941, 5, 6941], …]
        length (_type_): 每个样本的字符数，用于校验是否可以对齐
        labels (_type_): 字符标签，形如 [0,5,102,10,…]
        feature_embedding (_type_): 输入特征图
        num_classes (_type_): 字典长度

    Returns:
        _type_: _description_
    """
        # """
    # """ 得到用于计算 centerloss 的 embedding features，和对应的标签

    # Args:
    #     pred_label (_type_): 原始的预测结果，形如 [[6941, 6941, 0, 6941, 6941, 5, 6941], …]
    #     length (_type_): 每个样本的字符数，用于校验是否可以对齐
    #     labels (_type_): 字符标签，形如 [0,5,102,10,…]
    #     feature_embedding (_type_): 输入特征图
    # """
    device = labels.device
    
    # 判断是否为预测的字符
    B, C, H, W = feature_embedding.shape
    embedding = feature_embedding.view(B, C, H* W )
    embedding = embedding.permute(0, 2, 1) # w,batorchh_size,c -> batorchh_size, w, c
    raw_pred = pred_label.permute(1, 0)#w, batorchhsize -> batorchhsize, w
    
    char_pos_return = torch.tensor([]).to(device)
    label_return = []
    batorchh_num_mark = torch.tensor(0)
    batorchh_num = {}
    i = 0
    for i in range(raw_pred.shape[0]):
        is_char = torch.le(raw_pred[i], num_classes - 1).to(device)
        # x = raw_pred[i][:-1].to(device)
        # b = raw_pred[i][1:].to(device)
        # char_rep = torch.eq(x, b).to(device)
        # tail = torch.gt(raw_pred[i][:1], num_classes - 1).to(device)
        # char_rep = torch.cat([char_rep, tail]).to(device)
        # remove zero whose value is true        
        mask = torch.le(raw_pred[i], 1)
        # mask_or = torch.logical_or(mask, char_rep)
        char_no_rep = torch.logical_and(is_char, torch.logical_not(mask).to(device)).to(device)
        #char_no_rep = torch.logical_and(is_char, torch.logical_not(char_rep).to(device)).to(device)
        char_pos = torch.nonzero(char_no_rep, as_tuple=False).to(device)
        label = labels[:length[i]]
        labels = labels[length[i]:]
        pre_len = char_pos.size()[0]
        if not torch.eq(torch.tensor(pre_len), length[i]):
            continue
        batorchh_i = torch.zeros_like(char_pos).to(device)
            
        batorchh_i = batorchh_i + i
        char_pos_index = torch.cat([batorchh_i, char_pos], 1).to(device)
        char_pos_return = torch.cat([char_pos_return, char_pos_index],0).to(device)
        #print("char pos return: ", char_pos_return.size())
        #label_return_tensor = torch.cat([label_return_tensor, label],1).to(device)
        #print("label_return_tensor: ", label_return_tensor)
        # 将可以预测出字符数和gt一致的label, pos保存起来
        label_return = label_return + label 
        batorchh_num_mark = batorchh_num_mark + torch.tensor(1)
        batorchh_num[i] = pre_len
    #print("-========================")
    # 根据字符位置得到字符的 embedding
    if batorchh_num_mark != 0:
        # char_pos_return: [[24, 42], [24, 46], [24, 49], [24, 52], [24, 53], [24, 55], [24, 60], [24, 62], [24, 66], [24, 69], [24, 74], [24, 78], [24, 117], [33, 52], [33, 55], [33, 57], [33, 60], [33, 63], [33, 117], [35, 38], [35, 43], [35, 47], [35, 52], [35, 57], [35, 62], [35, 67], [35, 76], [35, 81], [35, 117], [49, 38], [49, 43], [49, 47], [49, 52], [49, 57], [49, 67], [49, 71], [49, 76], [49, 81], [49, 117], [72, 43], [72, 51], [72, 53], [72, 57], [72, 61], [72, 64], [72, 69], [72, 72], [72, 77], [72, 117] #  其中24表示batorchh中第24张图，42表示slice42 
        # label_return: [tensor([7165, 7246, 7221, 7225, 7166, 7209, 7164, 7187, 7182, 7246, 7166, 7204,7182], dtype=torch.int32), tensor([7182, 7184, 7163, 7164, 7246, 7182], dtype=torch.int32), tensor([5788,  512, 5182, 4295,  443, 5054, 5180,  186, 3120, 5205],
        embedding, label_tensor = get_features(batorchh_num, char_pos_return, embedding, label_return)
        return embedding, label_tensor #label_return是一维的数据
    else:
        return None, None
    
