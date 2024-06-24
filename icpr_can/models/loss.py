''' 自定义损失函数

    作者：jinqiu
    创建日期：2022/11/02
'''
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing. """

    def __init__(self, reduce=None,reduction='mean', smoothing=0.1):
        """ Constructor for the LabelSmoothing module. :param smoothing: label smoothing factor """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.reduce = reduce
        self.reduction = reduction
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.reduce:
            return loss.mean()
        else:
            return loss
        
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx=0, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target,T = 4):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        true_dist = F.softmax(true_dist,dim=-1)
        x = F.log_softmax(x, dim=-1)
        return self.criterion(x, Variable(true_dist, requires_grad=False))*T*T

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
    
    
class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True,checkpoint=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        
        if checkpoint is not None:
            assert os.path.exists(checkpoint), \
            f"center path({checkpoint}) must exist when it is not None."
            with open(checkpoint, 'rb') as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    self.centers[key] = torch.to_tensor(char_dict[key])


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size*num_classes, feat_dim).
            labels: ground truth labels with shape (batch_size*num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        batch_size = x.size(0)

        x= x.to(self.centers.device)
        labels = labels.to(self.centers.device)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat.addmm_(x,self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
       
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))


        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss
    
class ACELoss(nn.Module):
    def __init__(self, alpha=0, size_average=False):
        super(ACELoss, self).__init__()
        self.alpha = alpha
        self.size_average = size_average
        
    def one_hot(self,tensors, num_classes):
        onehot = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(1)
            t = torch.zeros(tensor.shape[0], num_classes).scatter_(1, tensor, 1)
            onehot.append(t)
        onehot = torch.stack(onehot)
        return onehot

    def forward(self, logits, targets, input_lengths, target_lengths):
        T_, B, C = logits.size()

        tagets_split = list(torch.split(targets, target_lengths.tolist()))
        targets_padded = torch.nn.utils.rnn.pad_sequence(tagets_split, batch_first=True, padding_value=0)
        targets_padded = self.one_hot(targets_padded.long(), num_classes=C)

        targets_padded = (targets_padded * (1-self.alpha)) + (self.alpha/C)
        targets_padded = torch.sum(targets_padded, 1).float()
        targets_padded[:,0] = T_ - target_lengths

        probs = torch.softmax(logits, dim=2)
        probs = torch.sum(probs, 0)
        probs = probs / T_
        targets_padded = targets_padded / T_
        #targets_padded = F.normalize(targets_padded, p=1, dim=1)
        #loss = F.kl_div(torch.log(probs), targets_padded, reduction='sum')

        #print(-torch.sum(torch.log(probs[0]) * targets_padded[0])) , (-torch.sum(torch.log(probs[1:]) * targets_padded[1:]))
        loss = -torch.sum(torch.log(probs) * targets_padded) / B
        return loss
    
class AngularPenaltySMLoss(nn.Module):

    def __init__(self, feat_dim, num_classes, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = feat_dim
        self.out_features = num_classes
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)



if __name__ == "__main__":
    feature_output = torch.tensor([[[0.2, 0.2, 0.6], [0.1, 0.1, 0.8]]]) # [W*H,B,C]
    output = torch.tensor([[0.2, 0.2, 0.6], [0.1, 0.1, 0.8]]) # [B,C]
    target = torch.tensor([1, 2]) # [B,L]
    input_lengths, target_lengths = torch.IntTensor([2]),torch.IntTensor([2]) #[B,1]
    ce_loss = nn.CrossEntropyLoss()
    sce_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    ls_loss = LabelSmoothing(size=3,smoothing=0.1)
    focal_loss = FocalLoss()
    center_loss = CenterLoss(num_classes=3,feat_dim=3,use_gpu=False)
    ace_loss = ACELoss()
    arcface_loss = AngularPenaltySMLoss(feat_dim=3,num_classes=3)
    print("CE: ",ce_loss(output, target))
    print("smooth CE: ",sce_loss(output, target).mean())
    print("label smooth: ",ls_loss(output, target).mean())
    print("focal: ",focal_loss(output, target))
    print("ace loss: ",ace_loss(feature_output,target,input_lengths, target_lengths))
    print("center loss: ",center_loss(output,target))
    print("arcface loss: ",arcface_loss(output,target))