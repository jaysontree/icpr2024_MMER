
import torch
import numpy as np
import torch.nn as nn

from functools import reduce
from numpy import unravel_index, ravel_multi_index

from pdb import set_trace as bp

def prev_seq(idx, k):
    """ Previous element of the sequence in a specific direction. """
    idx_bis = list(idx)
    idx_bis[k] -= 1
    idx_bis = tuple(idx_bis)
    return idx_bis

def prev_seq_flat(idx, k, shape):
    idx_bis = list(idx)
    idx_bis[k] -= 1
    idx_bis = ravel_multi_index(idx_bis, shape)
    return idx_bis

def prev_all(a, idx, shape_idx, shape_t):
    def default(): return torch.zeros(shape_t)
    return [a[prev_seq_flat(idx, k, shape_idx)] if idx[k] else default() for k in range(len(idx))]

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, B, hidden = recurrent.size()
        t_rec = recurrent.view(T * B, hidden)
        output = self.embedding(t_rec)
        output = output.view(T, B, -1)
        return output
    
    
class BidirectionalGRU(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, B, hidden = recurrent.size()
        t_rec = recurrent.view(T * B, hidden)
        output = self.embedding(t_rec)
        output = output.view(T, B, -1)
        return output
    
    
class MDLSTM(nn.Module):
    """ Applies a single-layer LSTM to an input multi-dimensional sequence.
    Args:
        size_in: The number of expected features in the input `x`.
        dim_in: Dimensionality of the input (e.g. 2 for images).
        size_out: The number of features in the hidden state `h`.
    Input: x
        - **x** of shape `(d1, ..., dn, batch, size_in)`: tensor containing input features.
    Output: h
        - **h** of shape `(d1, ..., dn, batch, size_out)`: tensor containing output features.
    """
    def __init__(self, size_in, dim_in, size_out):
        super().__init__()
        self.mdlstmcell = MDLSTMCell(size_in, dim_in, size_out)
        self.size_out = size_out
    def iter_idx(self, dimensions):
        # Multi-dimensional range().
        # Simply uses NumPy's unravel_index() function to get from an integer
        # index to a tuple index.
        idx_total = reduce(lambda a,b: a*b, dimensions)
        x = 0
        while x < idx_total:
            yield unravel_index(x, dimensions)
            x += 1
    def forward(self, x):
        # **Note on states:**
        # States are stored as "1d"-tensors (bare for the batch size and output
        # dimension) and concatenated.
        # Uses unravel_index() and ravel_multi_index() to go from 1d indexing
        # to multi-dimensional indexing.
        # Ideally, s and h would be pre-allocated with shape `(d1, ..., dn,
        # batch_size, self.size_out)` and filled with a loop, but we CANNOT do
        # this. Filling the tensors with a loop would be doing in-place
        # operations, and PyTorch does not like in-place operations, even though
        # in this specific case this should cause no issue.
        shape_idx = x.shape[:-2]
        batch_size = x.shape[-2]
        shape_t = (batch_size, self.size_out)
        s = []
        h = []
        for idx in self.iter_idx(shape_idx):
            s_new, h_new = self.mdlstmcell(x[idx], (prev_all(s, idx, shape_idx, shape_t), prev_all(h, idx, shape_idx, shape_t)))
            s.append(s_new.reshape((1, *shape_t)))
            h.append(h_new.reshape((1, *shape_t)))
        h = torch.cat(h)
        h = torch.reshape(h, (*shape_idx, *shape_t))
        return h
    
    
class MDLSTMCell(torch.nn.Module):
    """ Single cell of a multi-dimensional LSTM.
    Args:
        size_in: The number of expected features in the input `x`.
        dim_in: Dimensionality of the input (e.g. 2 for images).
        size_out: The number of features in the hidden state `h`.
    Input: x, (s_0,h_0)
        - **x** of shape `(batch, size_in)`: tensor containing input features.
        - **s_0** of shape `(batch, size_in)`: tensor containing the initial cell state for each element in the batch.
        - **h_0** of shape `(batch, size_in)`: tensor containing the initial hidden state for each element in the batch.
    Outputs: (s, h)
        - **s** of shape `(batch, size_out)`: tensor containing the next cell state for each element in the batch.
        - **h** of shape `(batch, size_out)`: tensor containing the next hidden state for each element in the batch.
    """
    def __init__(self, size_in, dim_in, size_out):
        super().__init__()
        self.size_in  = size_in # Number of input features
        self.dim_in   = dim_in # Dimensionality of the input (2 for images for instance).
        self.size_out = size_out # Number of output features
        # Parameters:
        # - Forget gates
        self.wf = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_in, self.size_out))
        self.uf = torch.nn.Parameter(torch.Tensor(self.dim_in, self.dim_in, self.size_out))
        self.biasf = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        # - Input gate
        self.wi = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.ui = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        self.biasi = torch.nn.Parameter(torch.Tensor(self.size_out))
        # - Output gate
        self.wo = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.uo = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        self.biaso = torch.nn.Parameter(torch.Tensor(self.size_out))
        # - Cell input gate
        self.wc = torch.nn.Parameter(torch.Tensor(self.size_in, self.size_out))
        self.uc = torch.nn.Parameter(torch.Tensor(self.dim_in, self.size_out))
        self.biasc = torch.nn.Parameter(torch.Tensor(self.size_out))
        # Initialize weights
        k = np.sqrt(1/size_out)
        torch.nn.init.uniform_(self.wf, a=-k, b=k)
        torch.nn.init.uniform_(self.uf, a=-k, b=k)
        torch.nn.init.uniform_(self.biasf, a=-k, b=k)
        torch.nn.init.uniform_(self.wi, a=-k, b=k)
        torch.nn.init.uniform_(self.ui, a=-k, b=k)
        torch.nn.init.uniform_(self.biasi, a=-k, b=k)
        torch.nn.init.uniform_(self.wo, a=-k, b=k)
        torch.nn.init.uniform_(self.uo, a=-k, b=k)
        torch.nn.init.uniform_(self.biaso, a=-k, b=k)
        torch.nn.init.uniform_(self.wc, a=-k, b=k)
        torch.nn.init.uniform_(self.uc, a=-k, b=k)
        torch.nn.init.uniform_(self.biasc, a=-k, b=k)
    def forward(self, x, old):
        device = x.device
        s_0, h_0 = old
        # Note on input dimension:
        # - x is of size (batch_size, self.size_in). It is the value of the
        #   input sequence at a given position.
        # - s_0 and h_0 are of size (self.dim_in,batch_size,self.size_out). They
        #   are the values of the cell state and hidden state at the "previous"
        #   positions, previous from every dimension (default should be 0).
        # 1/ Forget, input, output and cell activation gates.
        f = [torch.sigmoid(self.biasf[l] + torch.mm(x, self.wf[l]) + sum(torch.mul(h_0[k], self.uf[l][k]) for k in range(self.dim_in))) for l in range(self.dim_in)]
        i = torch.sigmoid(self.biasi + torch.mm(x, self.wi) + sum(torch.mul(h_0[k], self.ui[k]) for k in range(self.dim_in)))
        o = torch.sigmoid(self.biaso + torch.mm(x, self.wo) + sum(torch.mul(h_0[k], self.uo[k]) for k in range(self.dim_in)))
        c = torch.sigmoid(self.biasc + torch.mm(x, self.wc) + sum(torch.mul(h_0[k], self.uc[k]) for k in range(self.dim_in)))

        # 2/ Cell state
        s = torch.mul(i, c) + sum(torch.mul(f[k], s_0[k]) for k in range(self.dim_in))
        # 3/ Final output
        h = torch.mul(o, torch.tanh(s))
        
        return (s, h)