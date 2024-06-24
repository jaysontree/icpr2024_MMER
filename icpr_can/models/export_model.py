import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from models.densenet import DenseNet
from models.attention import Attention
from models.decoder import PositionEmbeddingSine
from models.counting import MultiCountingDecoder
from counting_utils import gen_counting_label
from utils import draw_attention_map, draw_counting_map


class ExportModel(nn.Module):

    def __init__(self, params=None):
        super(ExportModel, self).__init__()
        self.params = params
        self.device = params['device']
        self.encoder = EncoderModel(params=self.params)
        self.init_model = InitModel(params=self.params)
        self.decoder = DecoderModel(params=self.params)

    def forward(self, images, is_train=False):
        # 特征提取
        cnn_features, counting_preds = self.encoder(images,is_train=is_train)
        
        # 初始化
        batch_size, _, height, width = cnn_features.shape
        image_mask = torch.ones((batch_size, 1, height, width)).to(self.device)
        hidden, counting_feature , cnn_features_trans = \
            self.init_model(cnn_features, image_mask,counting_preds,is_train=is_train)
        
        # 解码
        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device) # 注意力覆盖总和
        word = torch.ones([batch_size]).long().to(device=self.device) #初始值
        
        i = 0
        word_probs = []
        word_alphas = []
        while i < 200:
            word_prob, word_alpha,word_alpha_sum,hidden = self.decoder(word,cnn_features,image_mask,cnn_features_trans,
                                                   counting_feature,hidden,word_alpha_sum,is_train=is_train)
            
            _, word = word_prob.max(1)
            
            if word.item() == 0:
                return word_probs, word_alphas
            word_alphas.append(word_alpha)
            word_probs.append(word.item())
            i += 1
        return word_probs, word_alphas



class EncoderModel(nn.Module):

    def __init__(self, params) -> None:
        super(EncoderModel, self).__init__()
        self.params = params
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']

        self.encoder = DenseNet(params=self.params)
        self.counting_decoder = MultiCountingDecoder(self.in_channel,
                                                     self.out_channel)

    def forward(self, x, mask=None, is_train=False):
        cnn_features = self.encoder(x)
        counting_preds, _, _ = self.counting_decoder(cnn_features, None)
        return cnn_features, counting_preds


class InitModel(nn.Module):
    def __init__(self, params) -> None:
        super(InitModel, self).__init__()
        self.params = params
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.word_num = params['word_num']

        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.position_embedding = PositionEmbeddingSine(256, normalize=True)
        self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim, kernel_size=1)
        self.counting_context_weight = nn.Linear(self.word_num, self.hidden_size)

    def forward(self,cnn_features,image_mask,counting_preds,is_train=False):
        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        pos = self.position_embedding(cnn_features_trans, image_mask[:,0, :, :])
        cnn_features_trans = cnn_features_trans + pos
        counting_context_weighted = self.counting_context_weight(counting_preds)
        
        hidden = self.init_hidden(cnn_features, image_mask)
        
        return hidden, counting_context_weighted, cnn_features_trans

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)


class DecoderModel(nn.Module):

    def __init__(self, params) -> None:
        super(DecoderModel, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.word_num = params['word_num']

        self.embedding = nn.Embedding(self.word_num, self.input_size) # 输入label的embedding
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size) # GRU
        
        # 计算预测值时的维度变换
        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size,self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel,self.hidden_size)
        
        self.word_convert = nn.Linear(self.hidden_size, self.word_num) # 计算最终的预测
        
        self.word_attention = Attention(params) # 计算注意力
        
        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def forward(self,word,cnn_features,image_mask,cnn_features_trans,
                counting_context_weighted,hidden, word_alpha_sum, is_train=False):
        word_embedding = self.embedding(word)
        hidden = self.word_input_gru(word_embedding, hidden)
        word_context_vec, word_alpha, word_alpha_sum = self.word_attention(
            cnn_features, cnn_features_trans, hidden, word_alpha_sum,
            image_mask)

        current_state = self.word_state_weight(hidden)
        word_weighted_embedding = self.word_embedding_weight(word_embedding)
        word_context_weighted = self.word_context_weight(word_context_vec)

        if self.params['dropout']:
            word_out_state = self.dropout(current_state +
                                          word_weighted_embedding +
                                          word_context_weighted +
                                          counting_context_weighted)
        else:
            word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

        word_prob = self.word_convert(word_out_state)

        return word_prob, word_alpha, word_alpha_sum, hidden
