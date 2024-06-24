import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

INIT = 1e-2


def refine_dict(srcdict, savekeys):
    _dict = OrderedDict()
    for k, v in srcdict.items():
        if k in savekeys:
            _dict[k] = v
    return _dict


class LatexInitDecoder(nn.Module):

    def __init__(self, dec_rnn_h=256, enc_out_dim=256):
        super(LatexInitDecoder, self).__init__()
        self.init_wh = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)

    def _init_h(self, mean_enc_out):
        return torch.tanh(self.init_wh(mean_enc_out))

    def _init_c(self, mean_enc_out):
        return torch.tanh(self.init_wc(mean_enc_out))

    def _init_o(self, mean_enc_out):
        return torch.tanh(self.init_wo(mean_enc_out))

    def forward(self, enc_out):
        # B H*W 256 to B 256
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        o = self._init_o(mean_enc_out)
        # h c o —— B 256
        # h and c tuple for rnncell
        return (h, c), o


class LatexDecoderHeader(nn.Module):

    def __init__(self,
                 out_size=172,
                 emb_size=128,
                 dec_rnn_h=256,
                 enc_out_dim=256):
        super(LatexDecoderHeader, self).__init__()
        self.rnn_decoder = nn.LSTMCell(dec_rnn_h + emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)
        self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)
        self.W_3 = nn.Linear(dec_rnn_h + enc_out_dim, dec_rnn_h, bias=False)
        self.W_out = nn.Linear(dec_rnn_h, out_size, bias=False)
        # beta is attention module weight......
        self.beta = nn.Parameter(torch.Tensor(enc_out_dim), requires_grad=True)
        init.uniform_(self.beta, -INIT, INIT)

    # 获取 attention，注意 attention 的形状应该是怎样的
    def _get_attn(self, enc_out, h_t):
        # cal alpha
        # enc out is B H*W 256 to B H*W 256
        alpha = torch.tanh(self.W_1(enc_out) + self.W_2(h_t).unsqueeze(1))
        alpha = torch.sum(self.beta * alpha, dim=-1)  # [B, L]
        alpha = F.softmax(alpha, dim=-1)  # [B, L]
        # cal context: [B, C]
        context = torch.bmm(alpha.unsqueeze(1), enc_out)
        context = context.squeeze(1)
        return context, alpha

    def forward(self, dec_states, o_t, enc_out, lasttarget):
        # lasttarget size B * num_class
        prev_y = self.embedding(lasttarget).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, o_t], dim=1)  # [B, emb_size+dec_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t:[B, dec_rnn_h]
        # context_t : [B, C]
        context_t, attn_scores = self._get_attn(enc_out, h_t)
        # [B, dec_rnn_h]
        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        # calculate logit
        logit = F.softmax(self.W_out(o_t), dim=1)  # [B, out_size]
        return (h_t, c_t), o_t, logit


class LatexDecoder(nn.Module):

    def __init__(self,params):
        super(LatexDecoder, self).__init__()
        self.max_len = 200
        enc_out_dim = params['encoder']['out_channel']
        out_size = self.word_num = params['word_num']
        dec_rnn_h = 256
        emb_size=256
        enc_out_dim=256
        self.decoderinit = LatexInitDecoder(dec_rnn_h=dec_rnn_h,enc_out_dim=enc_out_dim)
        self.decoder = LatexDecoderHeader( out_size=out_size, emb_size=emb_size,
                                          dec_rnn_h= dec_rnn_h, enc_out_dim=enc_out_dim)

    def forward_train(self, imgs, target_list):
        target_len = target_list.data.shape[1]
        state, o = self.decoderinit(imgs)
        # set the decode size ......
        logits = []
        for t in range(target_len):
            target = target_list[:, t]
            state, o, logit = self.decoder(state, o, imgs, target)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        return logits

    def forward_test(self, imgs):
        B, _, _, _ = imgs.data.shape
        imgs = self.encoder(imgs)

        state, o = self.decoderinit(imgs)
        tgt = torch.ones(B, 1).long() * 171
        formulas_idx = torch.ones(B, self.max_len).long() * 0
        for t in range(self.max_len):
            state, o, logit = self.decoder(state, o, imgs, tgt)
            tgt = torch.argmax(logit, dim=1, keepdim=True)
            # print(tgt)
            # if tgt == 0:
            #    break
            formulas_idx[:, t:t + 1] = tgt
        return formulas_idx

    def forward(self, imgs, target_list=None):
        if target_list is None:
            ret = self.forward_test(imgs)
        else:
            ret = self.forward_train(imgs, target_list)
        return ret



if __name__ == "__main__":
    model = LatexDecoder()
    a = torch.randn(1, 1, 32, 320)
    print(model(a).shape)
