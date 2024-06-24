import os
import time
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp

from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_crohme_dataset
from models.can import CAN
from training import train, eval

parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--config_file', default='config.yaml', type=str, help='配置文件')
parser.add_argument('--check', action='store_true', help='测试代码选项')
# DDP模式相关参数设置
parser.add_argument('--use_amp', type=bool, default=False)
parser.add_argument('--distributed', type=bool, default=True)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
args = parser.parse_args()

"""加载config文件"""
params = load_config(args.config_file)
params['distributed'] = args.distributed # 是否使用DDP模型，默认DP模式
params['use_amp'] = args.use_amp # 是否使用混合精度训练


"""设置随机种子"""
if args.distributed:
    random.seed(params['seed']+args.local_rank)
    np.random.seed(params['seed']+args.local_rank)
    torch.manual_seed(params['seed']+args.local_rank)
    torch.cuda.manual_seed(params['seed']+args.local_rank)
    torch.cuda.manual_seed_all(params['seed']+args.local_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""设置GPU"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
if args.distributed:
    """DDP初始化"""
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    device = torch.device("cuda", local_rank)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

"""设置使用混合精度"""
if args.use_amp:
    scaler = GradScaler()
    params['scaler'] = scaler


"""加载数据"""
if args.dataset == 'CROHME':
    train_loader, eval_loader, train_sampler = get_crohme_dataset(params)

"""加载模型"""
model = CAN(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'
print(model.name)

"""设置模型并行"""
gpu_number = torch.cuda.device_count()
if gpu_number>1:
    multi_gpu_flag = True
else:
    multi_gpu_flag = False
device_ids = [i for i in range(gpu_number)]
if multi_gpu_flag:
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = ddp(model, device_ids=[local_rank],output_device=local_rank)
    else:
        model = nn.DataParallel(model, device_ids=device_ids)

mdoel = model.to(device=device)

if gpu_number > 0:
    model.cuda()

"""启动tensorboard"""
if args.check:
    writer = None
else:
    if multi_gpu_flag:
        writer = SummaryWriter(f'{params["log_dir"]}/{model.module.name}')
    else:
        writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')
        

"""设置优化器"""
# filter(lambda p: p.requires_grad, model.parameters())
# model.parameters()
optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), 
                                                      lr=float(params['lr']),
                                                      eps=float(params['eps']), 
                                                      weight_decay=float(params['weight_decay']))

"""加载预训练模型"""
if params['finetune']:
    print('加载预训练模型权重')
    print(f'预训练权重路径: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'], skip_missmatch= True)

"""设置模型保存节点"""
if not args.check:
    if multi_gpu_flag:
        if not os.path.exists(os.path.join(params['checkpoint_dir'], model.module.name)):
            os.makedirs(os.path.join(params['checkpoint_dir'], model.module.name), exist_ok=True)
        os.system(f'cp {args.config_file} {os.path.join(params["checkpoint_dir"], model.module.name, model.module.name)}.yaml')
    else:
        if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
            os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
        os.system(f'cp {args.config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')

"""在CROHME上训练"""
if args.dataset == 'CROHME' or args.dataset == 'CoMER':
    min_score, init_epoch = 0, 115

    for epoch in range(init_epoch, params['epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader, writer=writer)

        if epoch >= params['valid_start']:
            eval_loss, eval_word_score, eval_exprate = eval(params, model, epoch, eval_loader, writer=writer)
            print(f'Epoch: {epoch+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
            if eval_exprate > params['best_score'] or (eval_exprate > min_score and not args.check and epoch >= params['save_start']):
                min_score = eval_exprate if eval_exprate > min_score else min_score
                if args.distributed:
                    if local_rank == 0:
                        save_checkpoint(model, optimizer, eval_word_score, eval_exprate, epoch+1,
                                        optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
                else:
                    save_checkpoint(model, optimizer, eval_word_score, eval_exprate, epoch+1,
                                    optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])