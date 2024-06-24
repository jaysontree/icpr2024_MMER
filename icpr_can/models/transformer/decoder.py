import sys
sys.path.append("/home/qiujin/project/CAN_src")

from typing import List
import math
import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from models.transformer.arm import AttentionRefinementModule
from models.transformer.transformer_decoder import TransformerDecoder,TransformerDecoderLayer

class ImgPosEnc(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        d_model: int = 512,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: float = None,
        device: int = None
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.device = device

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """add image positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Returns
        -------
        torch.Tensor
            [b, h, w, d]
        """
        not_mask = ~mask.ge(1)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            0, self.half_d_model, 2, dtype=torch.float, device=self.device
        )
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        pos_x = torch.einsum("b h w, d -> b h w d", x_embed, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", y_embed, inv_feq)

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3)

        x = x + pos
        return x

class WordPosEnc(nn.Module):
    def __init__(
        self, d_model: int = 512, max_len: int = 500, temperature: float = 10000.0
    ) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / d_model))

        inv_freq = torch.einsum("i, j -> i j", position, div_term)

        pe[:, 0::2] = inv_freq.sin()
        pe[:, 1::2] = inv_freq.cos()
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """add positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, l, d]

        Returns
        -------
        torch.Tensor
            [b, l, d]
        """
        _, seq_len, _ = x.size()
        emb = self.pe[:seq_len, :]
        x = x + emb[None, :, :]
        return x

def to_tgt_output(tokens,direction,device,pad_to_len = None):
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = SOS_IDX
        stop_w = EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = EOS_IDX
        stop_w = SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value= PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value= PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        token = [i+1 for i in token]
        tgt[i, 1 : (1 + lens[i])] = torch.tensor(token, dtype=torch.long)

        out[i, : lens[i]] = torch.tensor(token, dtype=torch.long)
        out[i, lens[i]] = stop_w

    return tgt, out

def to_bi_tgt_out(tokens: List[List[int]], device: torch.device):
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out



def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder


class TransDecoder(nn.Module):
    def __init__(self,params,):
        super().__init__()

        self.params = params
        self.vocab_size = params['word_num']
        self.device = params['device']
        self.pad_idx = params['transformer']['pad_idx']
        self.sos_idx = params['transformer']['sos_idx']
        self.eos_idx = params['transformer']['eos_idx']
        self.max_len = params['transformer']['max_len']

        self.d_model = params['transformer']['d_model']
        self.nhead = params['transformer']['nhead']
        self.num_decoder_layers = params['transformer']['num_decoder_layers']
        self.dim_feedforward = params['transformer']['dim_feedforward']
        self.dropout = params['transformer']['dropout']
        self.dc = params['transformer']['dc']
        self.cross_coverage = params['transformer']['cross_coverage']
        self.self_coverage = params['transformer']['self_coverage']
        
        self.feature_proj = nn.Conv2d(params['encoder']['out_channel'], self.d_model, kernel_size=1)
        self.pos_enc_2d = ImgPosEnc(self.d_model, normalize=True,device=self.device)
        self.norm = nn.LayerNorm(self.d_model)

        self.word_embed = nn.Sequential(
            nn.Embedding(self.vocab_size, self.d_model), nn.LayerNorm(self.d_model)
        )

        self.pos_enc = WordPosEnc(d_model=self.d_model)

        self.norm = nn.LayerNorm(self.d_model)

        self.model = _build_transformer_decoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            dc=self.dc,
            cross_coverage=self.cross_coverage,
            self_coverage=self.self_coverage,
        )

        self.proj = nn.Linear(self.d_model, self.vocab_size)

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor = None ,is_train:bool = True) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        ------
        FloatTensor
            [b, l, vocab_size]
        """
        if is_train:
            return self.forward_train(src=src,src_mask=src_mask,tgt=tgt)
        else:
            return self.forward_test(src=src,src_mask=src_mask,tgt=tgt)
    
    def forward_train(self,src: FloatTensor, src_mask: LongTensor, tgt: LongTensor) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        # convert dims
        src = self.feature_proj(src)
        src = rearrange(src, "b d h w -> b h w d")
        src = self.pos_enc_2d(src, src_mask) # positional encoding
        src = self.norm(src)
        
        # 1d -> 2d
        src = torch.cat((src, src), dim=0)  # [2b, h,w, d]
        src_mask = torch.cat((src_mask, src_mask), dim=0)  #[2b,h,w]

        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == self.pad_idx

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")


        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out
        
    
    def forward_test(self, src: FloatTensor, src_mask: LongTensor,tgt: LongTensor = None) -> FloatTensor:
        """ forward test
        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        num_steps = 0
        if tgt is not None:
            num_steps = tgt.shape[1]
        else:
            num_steps = self.max_len
        
        batch_size = src.shape[0] * 2
        l2r = torch.full(
                (batch_size // 2, 1),
                fill_value=self.sos_idx,
                dtype=torch.long,
                device=self.device,
            )
        r2l = torch.full(
            (batch_size // 2, 1),
            fill_value=self.eos_idx,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = torch.cat((l2r, r2l), dim=0)
        
        cur_len = 0
        word_probs = []
        for i in range(num_steps):
            next_token_logits = self.forward_train(src,src_mask,input_ids)
            _, input_ids = next_token_logits.max(-1)
            word_probs.append(next_token_logits)
            
            cur_len += 1
            
        return torch.cat(word_probs,dim=1)


if __name__ == "__main__":
    from utils import load_config
    params = load_config("../../config.yaml")
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    device = torch.device('cpu' )
    params['device'] = device
    params['word_num']=224+1
    model = TransDecoder(params)
    src = torch.randn(2,684,3,3)
    src_mask = torch.zeros([2,3,3]).long()
    tgt = [[4,5,6],[7,8]]
    tgt, out = to_bi_tgt_out(tgt, device) #[2b, l]
    out = model(src,src_mask,tgt,is_train=True)
    print(out.shape)
    print(out)
