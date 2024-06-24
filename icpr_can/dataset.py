import os
import cv2
import torch
import time
import math
import random
import numpy as np
import pickle as pkl
import joblib
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from augment import DataAug

MAX_H,MAX_W=800,160

class ScaleToLimitRange:
    __max_iter__ = 10

    def __init__(self,
                 w_lo = 16,
                 w_hi = 672,
                 h_lo = 16,
                 h_hi = 192) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def is_img_valid(self, img: np.ndarray):
        h, w, _ = img.shape
        c = False
        if self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi:
            c = True
        return c

    def pad_resize(self, img: np.ndarray):
        h, w, c = img.shape
        # height
        if h > self.h_hi:
            ratio_h = h / self.h_hi
        elif h < self.h_lo:
            ratio_h = -1
        else:
            ratio_h = 1
        # width
        if w > self.w_hi:
            ratio_w = w / self.w_hi
        elif w < self.w_lo:
            ratio_w =  -1
        else:
            ratio_w = 1
        if ratio_h == 1 and ratio_w == 1:
            return img

        # pad first
        if ratio_h == -1 or ratio_w == -1:
            pw = math.ceil((self.w_lo - w)/2.) if ratio_w == -1 else 0
            ph = math.ceil((self.h_lo - h)/2.) if ratio_h == -1 else 0
            img_pad = np.ones((h + ph*2, w + pw*2, c), dtype=np.uint8) * 255
            img_pad[ph:ph+h, pw:pw+w,:] = img
            img = img_pad.copy()
        if ratio_h > 1 or ratio_w > 1:
            h, w,_ = img.shape
            ratio = max(ratio_h, ratio_w)
            img = cv2.resize(img, None, fx=1/ratio, fy=1/ratio, interpolation=cv2.INTER_LINEAR)

        return img

    def __call__(self, img: np.ndarray) -> np.ndarray:
        it = 0
        while not self.is_img_valid(img) and it < self.__max_iter__:
           img = self.pad_resize(img)
           it += 1
        
        return img



class ScaleAugmentation:
    def __init__(self, lo: float = 0.5 , hi: float = 1.4) -> None:
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        h, w, c = img.shape
        scale_img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        h, w, c = scale_img.shape
        return scale_img
    
class RandomPadding(object):
    def __init__(self, max_padding = 10, probability=0.5) -> None:
        self.max_padding = max_padding
        self.probability = probability
        
    def get_left_right_up_down(self, x, y):
        l, r, u, d = 0, len(x), 0, len(y)
        for i in range(len(x)):
            if x[i] > 0:
                l = i-1
                break
            
        for i in range(len(x)-1, -1, -1):
            if x[i] > 0:
                r = i+1
                break
            
        for i in range(len(y)):
            if y[i] > 0:
                u = i-1
                break
            
        for i in range(len(y)-1, -1, -1):
            if y[i] > 0:
                d = i+1
                break
        return l, len(x) - r, u, len(y) - d
    
    def random_padding(self, img):
        h, w, c = img.shape
        pad_img = np.ones((h+self.max_padding*2, w+self.max_padding*2, c), dtype= img.dtype)*255
        pad_img[self.max_padding:h+self.max_padding, self.max_padding:self.max_padding+w, :] = img
        gray_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        
        x_hist = np.sum(bin_img/255, axis=0).tolist()
        y_hist = np.sum(bin_img/255, axis=1).tolist()
        
        l, r, u, d = self.get_left_right_up_down(x_hist, y_hist)
        
        h, w, c = pad_img.shape
        l_offset = random.randint(0, l)
        x0 = l_offset
        r_offset = random.randint(0, r)
        x1 = w - r_offset
        u_offset = random.randint(0, u)
        y0 = u_offset
        d_offset = random.randint(0, d)
        y1 = h - d_offset
        
        if y1-y0 < 16 or x1-x0 < 16:
            return pad_img
        
        return pad_img[y0:y1,x0:x1,:]
    
    def __call__(self, img):
        if np.random.random() < self.probability:
            img = self.random_padding(img)
        return img 

class SynthrticAugmentation:
    def __init__(self, background, probability=0.1):
        self.probability = probability
        self.bg = self.load_background(background)
        self.bg_num = len(self.bg.keys())
        
    def load_background(slef, background_path):
        background = {}
        for idx, file in enumerate(os.listdir(background_path)):
            img_file = os.path.join(background_path, file)
            img = cv2.imread(img_file)
            background[idx] = img
        return background
    
    def crop_roi(self,bg_img,img):
        h,w,_ = img.shape
        bh,bw,_ = bg_img.shape
        
        h_ratio = bh/h
        w_ratio = bw/w
        ratio = min(w_ratio,h_ratio)
        # 如果背景图像小于当前图片
        if ratio < 1 :
            bg_img = cv2.resize(bg_img,None,fx=1/ratio,fy=1/ratio,interpolation=cv2.INTER_CUBIC)
            
        bh,bw,_ = bg_img.shape
        offset_y = random.randint(0, bh-h)
        offset_x = random.randint(0, bw-w)
        
        roi = bg_img[offset_y:offset_y+h,offset_x:offset_x+w,:]
        
        return roi.copy()
    
    def refine_fg(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,inv_gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        # kernel = np.ones((3,3),np.uint8)
        # inv_gray = cv2.morphologyEx(inv_gray,cv2.MORPH_DILATE,kernel)
        return inv_gray
    
    def replace_bg(self,img):
        """ 背景替换生成数据
        Args:
            img (_type_): _description_
        """
        # 随机选择一个背景
        bg_id = random.randint(0,self.bg_num-1)
        bg_img = self.bg[bg_id]
        # 随机选择背景区域
        bg_roi = self.crop_roi(bg_img,img)
        # 提取前景mask
        mask = self.refine_fg(img).astype(np.uint8)
        img_mask = cv2.bitwise_and(img,img,mask=mask)
        bg_mask = cv2.bitwise_and(bg_roi,bg_roi,mask=(255-mask))
        bg_mask = cv2.add(bg_mask, img_mask)
        syn_img = bg_mask
        return syn_img
    
    def __call__(self, img):
        if np.random.random() < self.probability:
            img = self.replace_bg(img)
            
        return img
        
class RandomPadding(object):
    def __init__(self, max_padding = 10, probability=1) -> None:
        self.max_padding = max_padding
        self.probability = probability
        
    def get_left_right_up_down(self, x, y):
        l, r, u, d = 0, len(x), 0, len(y)
        for i in range(len(x)):
            if x[i] > 0:
                l = i-1
                break
            
        for i in range(len(x)-1, -1, -1):
            if x[i] > 0:
                r = i+1
                break
            
        for i in range(len(y)):
            if y[i] > 0:
                u = i-1
                break
            
        for i in range(len(y)-1, -1, -1):
            if y[i] > 0:
                d = i+1
                break
        return l, len(x) - r, u, len(y) - d
    
    def random_padding(self, img):
        h, w, c = img.shape
        pad_img = np.ones((h+self.max_padding*2, w+self.max_padding*2, c), dtype= img.dtype)*255
        pad_img[self.max_padding:h+self.max_padding, self.max_padding:self.max_padding+w, :] = img
        gray_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        
        x_hist = np.sum(bin_img/255, axis=0).tolist()
        y_hist = np.sum(bin_img/255, axis=1).tolist()
        
        l, r, u, d = self.get_left_right_up_down(x_hist, y_hist)
        
        h, w, c = pad_img.shape
        l_offset = random.randint(0, l)
        x0 = l_offset
        r_offset = random.randint(0, r)
        x1 = w - r_offset
        u_offset = random.randint(0, u)
        y0 = u_offset
        d_offset = random.randint(0, d)
        y1 = h - d_offset
        return pad_img[y0:y1,x0:x1,:]
    
    def __call__(self, img):
        if np.random.random() < self.probability:
            img = self.random_padding(img)
        return img   


class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = joblib.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params
        self.data_aug = DataAug().aug_img
        self.scale_img = ScaleToLimitRange()
        self.scale_aug = ScaleAugmentation()
        self.synthrtic_aug = SynthrticAugmentation(background="/home/qiujin/data/priting_data/background")
        self.random_padding = RandomPadding()

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, labels = self.labels[idx].strip().split("\t")
        labels = labels.strip().split(' ')
        image = self.images[name]
        if self.is_train:
            image = self.random_padding(image)
            image = self.synthrtic_aug(image)
            image = self.scale_aug(image)
            image = self.data_aug(image)
        image = self.scale_img(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = torch.Tensor(image) / 255
        image = image.unsqueeze(0)
        labels.append('0')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        return image, words

class HMERDatasetV2(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HMERDatasetV2, self).__init__()
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        self.labels = []
        for lab in labels:
            if len(lab.strip().split("\t")) != 2:
                continue
                
            name, labels = lab.strip().split("\t")
            if not os.path.exists(os.path.join(image_path, name)):
                continue
            self.labels.append(lab)

        self.image_path = image_path
        self.words = words
        self.is_train = is_train
        self.params = params
        self.data_aug = DataAug().aug_img
        self.scale_img = ScaleToLimitRange()
        self.scale_aug = ScaleAugmentation()
        # self.random_padding = RandomPadding()
        # self.synthrtic_aug = SynthrticAugmentation(background="/home/qiujin/data/priting_data/background")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        name, labels = self.labels[idx].strip().split("\t")
        labels = labels.strip().split(' ')
        image_path = os.path.join(self.image_path, name)
        image = cv2.imread(image_path)
        if self.is_train:
            # image = self.random_padding(image)
            # image = self.synthrtic_aug(image)
            image = self.scale_aug(image)
            image = self.data_aug(image)
        image = self.scale_img(image)
        cv2.imwrite('tmp.jpg', image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = torch.Tensor(image) / 255
        image = image.unsqueeze(0)
        labels.append('<sos>')
        words = self.words.encode(labels)

        # print(words)
        # print(self.words.decode(words))
        words = torch.LongTensor(words)
        return image, words


def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    # print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    # print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")
    print(f"训练数据路径 images: {params['image_path']} labels: {params['train_path']}")
    print(f"验证数据路径 images: {params['image_path']} labels: {params['eval_path']}")

    # train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    # eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)
    train_dataset = HMERDatasetV2(params, params['image_path'], params['train_path'], words, is_train=True)
    eval_dataset = HMERDatasetV2(params, params['image_path'], params['eval_path'], words, is_train=False)

    if params['distributed']:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = RandomSampler(eval_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True,drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=params['batch_size'], sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True,drop_last=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader, train_sampler


def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length+1)).long(), torch.zeros((len(proper_items), max_length+1))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks



class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip().split("\t")[0]: i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip().split('\t')[0] for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        # label_index = [int(item) for item in labels]
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label

collate_fn_dict = {
    'collate_fn': collate_fn,
}


if __name__ == "__main__":

    scale_aug = ScaleAugmentation()
    data_aug = DataAug().aug_img
    scale_size = ScaleToLimitRange()

    im = cv2.imread("214653.png")
    im = scale_aug(im)
    im = data_aug(im)
    im = scale_size(im)

    cv2.imwrite("test.jpg",im)
    
