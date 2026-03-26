# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import collections
import copy
import numpy as np
from torch.optim import swa_utils
from tqdm import tqdm
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_dino, ft_net_convnext, ft_net_efficient, ft_net_NAS, PCB
from random_erasing import RandomErasing
from dgfolder import DGFolder
import yaml
from shutil import copyfile
from circle_loss import CircleLoss, convert_label_to_similarity
from instance_loss import InstanceLoss
from ODFA import ODFA
from utils import save_network
version =  torch.__version__
from pytorch_metric_learning import losses, miners #pip install pytorch-metric-learning

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
# data
parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--workers', default=0, type=int, help='dataloader workers, use 0 on macOS for compatibility')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--DG', action='store_true', help='use extra DG-Market Dataset for training. Please download it from https://github.com/NVlabs/DG-Net#dg-market.' )
# optimizer
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
parser.add_argument('--total_epoch', default=60, type=int, help='total training epoch')
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50%% memory' )
parser.add_argument('--bf16', action='store_true', help='use bfloat16 instead of float32, which will save about 50%% memory' )
parser.add_argument('--cosine', action='store_true', help='use cosine lrRate' )
parser.add_argument('--FSGD', action='store_true', help='use fused sgd, which will speed up trainig slightly. apex is needed.' )
parser.add_argument('--wa', action='store_true', help='use weight average' )
# backbone
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_swin', action='store_true', help='use swin transformer 224x224' )
parser.add_argument('--use_swinv2', action='store_true', help='use swin transformerv2' )
parser.add_argument('--use_dino', action='store_true', help='use dinov3' )
parser.add_argument('--use_efficient', action='store_true', help='use efficientnet-b4' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--use_hr', action='store_true', help='use hrNet' )
parser.add_argument('--use_convnext', action='store_true', help='use ConvNext' )
parser.add_argument('--ibn', action='store_true', help='use resnet+ibn' )
parser.add_argument('--usam', action='store_true', help='use resnet+usam (Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP2022)' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
# loss
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--cosface', action='store_true', help='use CosFace loss' )
parser.add_argument('--contrast', action='store_true', help='use contrast loss' )
parser.add_argument('--instance', action='store_true', help='use instance loss' )
parser.add_argument('--instance_id', action='store_true', help='use instance loss with ID' )
parser.add_argument('--ins_gamma', default=32, type=int, help='gamma for instance loss')
parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
parser.add_argument('--lifted', action='store_true', help='use lifted loss' )
parser.add_argument('--sphere', action='store_true', help='use sphere loss' )
parser.add_argument('--adv', default=0.0, type=float, help='add the adversarial training. Follow the paper `U-turn: Crafting Adversarial Queries with Opposite-direction Features, IJCV2022`' )
parser.add_argument('--aiter', default=10, type=float, help='enable adversarial loss every x iter' )

opt = parser.parse_args()

if opt.DG:
    opt.wa = True #DG will enable swa.

fp16 = opt.fp16
bf16 = opt.bf16
if fp16:
    dtype16 = torch.float16
elif bf16:
    dtype16 = torch.bfloat16

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)
opt.gpu_ids = gpu_ids
# set gpu ids
if len(gpu_ids)>0:
    #torch.cuda.set_device(gpu_ids[0])
    cudnn.enabled = True
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

if opt.use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(h, w),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

use_gpu = torch.cuda.is_available()
if torch.cuda.is_available() and len(opt.gpu_ids) > 0:
    device = torch.device('cuda:%d' % opt.gpu_ids[0])
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
use_amp = device.type == 'cuda' and (fp16 or bf16)
if (fp16 or bf16) and device.type != 'cuda':
    print('fp16/bf16 autocast is only enabled on CUDA in this repo. Running in full precision on %s.' % device.type)

loader_kwargs = dict(
    batch_size=opt.batchsize,
    shuffle=True,
    num_workers=opt.workers,
    pin_memory=(device.type == 'cuda'),
    drop_last=True,
)
if opt.workers > 0:
    loader_kwargs['prefetch_factor'] = 2
    loader_kwargs['persistent_workers'] = True

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], **loader_kwargs) # 8 workers may work faster
              for x in ['train', 'val']}

reid_eval_enabled = False
if os.path.isdir(os.path.join(data_dir, 'query')) and os.path.isdir(os.path.join(data_dir, 'gallery')):
    image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'), data_transforms['val'])
    image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'), data_transforms['val'])
    eval_loader_kwargs = dict(
        batch_size=max(1, opt.batchsize),
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=(device.type == 'cuda'),
    )
    if opt.workers > 0:
        eval_loader_kwargs['prefetch_factor'] = 2
        eval_loader_kwargs['persistent_workers'] = True
    dataloaders['query'] = torch.utils.data.DataLoader(image_datasets['query'], **eval_loader_kwargs)
    dataloaders['gallery'] = torch.utils.data.DataLoader(image_datasets['gallery'], **eval_loader_kwargs)
    reid_eval_enabled = True
# Use extra DG-Market Dataset for training. Please download it from https://github.com/NVlabs/DG-Net#dg-market.
if opt.DG:
    if not os.path.isdir('../DG-Market'):
        os.system('gdown 126Gn90Tzpk3zWp2c7OBYPKc-ZjhptKDo')
        os.system('unzip DG-Market.zip -d ../')
        os.system('rm DG-Market.zip')
        
    image_datasets['DG'] = DGFolder(os.path.join('../DG-Market' ),
                                          data_transforms['train'])
    dataloaders['DG'] = torch.utils.data.DataLoader(image_datasets['DG'], batch_size = max(8, opt.batchsize//2),
                                             shuffle=True, num_workers=opt.workers, drop_last=True, pin_memory=(device.type == 'cuda'))
    DGloader_iter = enumerate(dataloaders['DG'])

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1, device=img.device).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_id(img_path):
    camera_id = []
    labels = []
    for path, _ in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return np.array(camera_id), np.array(labels)

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    cmc[rows_good[0]:] = 1
    ngood = len(good_index)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc

def evaluate_rank(query_feature, query_label, query_cam, gallery_feature, gallery_label, gallery_cam):
    cmc = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    valid_queries = 0
    for i in range(len(query_label)):
        score = np.dot(gallery_feature, query_feature[i])
        index = np.argsort(score)[::-1]
        query_index = np.argwhere(gallery_label == query_label[i])
        camera_index = np.argwhere(gallery_cam == query_cam[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gallery_label == -1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1)
        ap_tmp, cmc_tmp = compute_mAP(index, good_index, junk_index)
        if cmc_tmp[0] == -1:
            continue
        cmc += cmc_tmp
        ap += ap_tmp
        valid_queries += 1

    if valid_queries == 0:
        return 0.0, 0.0

    cmc = cmc.float() / valid_queries
    return float(cmc[0]), float(ap / valid_queries)

def extract_embedding_batch(model, img):
    base_model = model.module if hasattr(model, 'module') else model

    if opt.PCB:
        x = base_model.model.conv1(img)
        x = base_model.model.bn1(x)
        x = base_model.model.relu(x)
        x = base_model.model.maxpool(x)
        x = base_model.model.layer1(x)
        x = base_model.model.layer2(x)
        x = base_model.model.layer3(x)
        x = base_model.model.layer4(x)
        x = base_model.avgpool(x)
        x = base_model.dropout(x)
        parts = []
        for i in range(base_model.part):
            parts.append(x[:, :, i].view(x.size(0), x.size(1)))
        return torch.cat(parts, dim=1)

    if opt.use_dense:
        x = base_model.model.features(img)
        x = x.view(x.size(0), x.size(1))
        return base_model.classifier.add_block(x)
    if opt.use_swin or opt.use_swinv2 or opt.use_dino:
        x = base_model.model.forward_features(img)
        if x.dim() == 3:
            x = base_model.avgpool1d(x.permute((0, 2, 1)))
        else:
            x = base_model.avgpool2d(x.permute((0, 3, 1, 2)))
        x = x.view(x.size(0), x.size(1))
        return base_model.classifier.add_block(x)
    if opt.use_convnext or opt.use_hr:
        x = base_model.model.forward_features(img)
        x = base_model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return base_model.classifier.add_block(x)
    if opt.use_efficient:
        x = base_model.model.extract_features(img)
        x = base_model.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return base_model.classifier.add_block(x)
    if opt.use_NAS:
        x = base_model.model.features(img)
        x = base_model.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        return base_model.classifier.add_block(x)

    x = base_model.model.conv1(img)
    x = base_model.model.bn1(x)
    x = base_model.model.relu(x)
    if getattr(base_model, 'usam', False):
        x = base_model.usam_1(x)
    x = base_model.model.maxpool(x)
    x = base_model.model.layer1(x)
    if getattr(base_model, 'usam', False):
        x = base_model.usam_2(x)
    x = base_model.model.layer2(x)
    x = base_model.model.layer3(x)
    x = base_model.model.layer4(x)
    x = base_model.model.avgpool(x)
    x = x.view(x.size(0), x.size(1))
    return base_model.classifier.add_block(x)

def extract_reid_feature(model, dataloader, linear_num):
    features = None
    for iter, data in enumerate(dataloader):
        img, _ = data
        n = img.size(0)
        ff = torch.zeros((n, linear_num), device=device, dtype=torch.float32)
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = img.to(device)
            ff += extract_embedding_batch(model, input_img)
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.detach().cpu()
        if features is None:
            features = torch.zeros((len(dataloader.dataset), ff.shape[1]), dtype=torch.float32)
        start = iter * dataloader.batch_size
        end = min((iter + 1) * dataloader.batch_size, len(dataloader.dataset))
        features[start:end, :] = ff[:end-start]
    return features

def run_reid_eval(model):
    if not reid_eval_enabled:
        return None

    model.eval()
    if opt.PCB:
        linear_num = 2048 * 6
    elif opt.linear_num > 0:
        linear_num = opt.linear_num
    elif opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
        linear_num = 1024
    elif opt.use_dino:
        linear_num = 768
    elif opt.use_efficient:
        linear_num = 1792
    elif opt.use_NAS:
        linear_num = 4032
    else:
        linear_num = 2048

    with torch.no_grad():
        gallery_feature = extract_reid_feature(model, dataloaders['gallery'], linear_num).numpy()
        query_feature = extract_reid_feature(model, dataloaders['query'], linear_num).numpy()

    gallery_cam, gallery_label = get_id(image_datasets['gallery'].imgs)
    query_cam, query_label = get_id(image_datasets['query'].imgs)
    rank1, mAP = evaluate_rank(query_feature, query_label, query_cam, gallery_feature, gallery_label, gallery_cam)
    return rank1, mAP

def train_model(model, criterion, optimizer, scheduler, scaler, num_epochs=25):
    since = time.time()
    last_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts = copy.deepcopy(model.state_dict())
    best_rank1 = -1.0
    best_map = -1.0
    best_epoch = -1

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    wa_flag = opt.wa
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch
    if opt.PCB:
        embedding_size = model.classifier0.linear_num
    else:
        embedding_size = model.classifier.linear_num
    if opt.arcface:
        criterion_arcface = losses.ArcFaceLoss(num_classes=opt.nclasses, embedding_size=embedding_size)
    if opt.cosface: 
        criterion_cosface = losses.CosFaceLoss(num_classes=opt.nclasses, embedding_size=embedding_size)
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32) # gamma = 64 may lead to a better result.
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    if opt.lifted:
        criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
    if opt.contrast: 
        criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if opt.instance or opt.instance_id:
        criterion_instance = InstanceLoss(gamma = opt.ins_gamma)
    if opt.sphere:
        criterion_sphere = losses.SphereFaceLoss(num_classes=opt.nclasses, embedding_size=embedding_size, margin=4)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        if opt.wa and wa_flag and epoch >=  num_epochs*0.8:
            wa_flag = False
            swa_model = swa_utils.AveragedModel(model)
            swa_model.avg_fn = swa_utils.get_ema_avg_fn(decay=0.996)
            print('start weight avg')
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            # Phases 'train' and 'val' are visualized in two separate progress bars
            pbar = tqdm()
            pbar.reset(total=len(dataloaders[phase].dataset))
            ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for iter, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                pbar.update(now_batch_size)  # update the pbar even in the last batch
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                inputs = inputs.to(device).detach()
                labels = labels.to(device).detach()
                # if we use low precision, input also need to be fp16
                #if fp16:
                #    inputs = inputs.half()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                elif use_amp:
                    with torch.amp.autocast(device_type='cuda',dtype=dtype16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if opt.adv>0 and iter%opt.aiter==0: 
                    inputs_adv = ODFA(model, inputs)
                    outputs_adv = model(inputs_adv)

                sm = nn.Softmax(dim=1)
                log_sm = nn.LogSoftmax(dim=1)
                return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere
                if return_feature: 
                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels) 
                    _, preds = torch.max(logits.data, 1)
                    if opt.adv>0  and iter%opt.aiter==0:
                        logits_adv, _ = outputs_adv
                        loss += opt.adv * criterion(logits_adv, labels)
                    if opt.arcface:
                        loss +=  criterion_arcface(ff, labels)/now_batch_size
                    if opt.cosface:
                        loss +=  criterion_cosface(ff, labels)/now_batch_size
                    if opt.circle:
                        loss +=  criterion_circle(*convert_label_to_similarity( ff, labels))/now_batch_size
                    if opt.triplet:
                        hard_pairs = miner(ff, labels)
                        loss +=  criterion_triplet(ff, labels, hard_pairs) #/now_batch_size
                    if opt.lifted:
                        loss +=  criterion_lifted(ff, labels) #/now_batch_size
                    if opt.contrast:
                        loss +=  criterion_contrast(ff, labels) #/now_batch_size
                    if opt.instance:
                        loss += criterion_instance(ff) /now_batch_size
                    if opt.instance_id:
                        loss += criterion_instance(ff, labels) /now_batch_size
                    if opt.sphere:
                        loss +=  criterion_sphere(ff, labels)/now_batch_size
                elif opt.PCB:  #  PCB
                    part = {}
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)
                else:  #  norm
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    if opt.adv>0 and iter%opt.aiter==0:
                        loss += opt.adv * criterion(outputs_adv, labels)

                del inputs
                # use extra DG Dataset (https://github.com/NVlabs/DG-Net#dg-market)
                if opt.DG and phase == 'train' and epoch > num_epochs*0.8:
                    # print("DG-Market is involved. It will double the training time.")
                    try:
                        _, batch = DGloader_iter.__next__()
                    except StopIteration: 
                        DGloader_iter = enumerate(dataloaders['DG'])
                        _, batch = DGloader_iter.__next__()
                    except UnboundLocalError:  # first iteration
                        DGloader_iter = enumerate(dataloaders['DG'])
                        _, batch = DGloader_iter.__next__()
                        
                    inputs1, inputs2, _ = batch
                    inputs1 = inputs1.to(device).detach()
                    inputs2 = inputs2.to(device).detach()
                    # use memory in vivo loss (https://arxiv.org/abs/1912.11164)
                    if use_amp:
                        with torch.amp.autocast(device_type='cuda', dtype=dtype16):
                            outputs1 = model(inputs1)
                    else:
                        outputs1 = model(inputs1)

                    if return_feature:
                        outputs1, _ = outputs1
                    elif opt.PCB:
                        for i in range(num_part):
                            part[i] = outputs1[i]
                        outputs1 = (sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5]))/6

                    swa_model.eval()
                    with torch.no_grad():
                        outputs2 = swa_model(inputs2) #stop gradient like dino

                    if return_feature:
                        outputs2, _ = outputs2
                    elif opt.PCB:
                        for i in range(num_part):
                            part[i] = outputs2[i]
                        outputs2 = (sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5]))/6

                    #supervised via teacher like dino. previous use sm(outputs1 + outputs2)
                    kl_loss = nn.KLDivLoss(reduction='batchmean')
                    reg= (kl_loss(log_sm(outputs2), sm(outputs1))  + kl_loss(log_sm(outputs1) , sm(outputs2)))/2
                    loss += 0.1*reg
                    del inputs1, inputs2
                    #print(0.01*reg)
                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss = loss*warm_up
                    print(loss, warm_up)

                if phase == 'train':
                    if bf16 or fp16: # we use optimier to backward loss
                        scaler.scale(loss).backward()
                        scaler.step(optimizer) # a safety optimizer.step()
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                    ordered_dict["Loss"] = f"{loss.item():.4f}"
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                    ordered_dict["Loss"] = f"{loss.data[0]:.4f}"
                del loss
                running_corrects += float(torch.sum(preds == labels.data))
                # Refresh the progress bar in every batch
                ordered_dict["phase"] = phase
                ordered_dict[
                    "Acc"
                ] = f"{(float(torch.sum(preds == labels.data)) / now_batch_size):.4f}"
                pbar.set_postfix(ordered_dict=ordered_dict)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            ordered_dict["phase"] = phase
            ordered_dict["Loss"] = f"{epoch_loss:.4f}"
            ordered_dict["Acc"] = f"{epoch_acc:.4f}"
            pbar.set_postfix(ordered_dict=ordered_dict)
            pbar.close()
            
            if phase == 'train' and opt.wa and epoch >= num_epochs*0.8: 
                swa_model.update_parameters(model)
                swa_utils.update_bn(dataloaders['train'], swa_model, device=device)

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            # deep copy the model
            if phase == 'val' and epoch%10 == 9:
                last_model_wts = copy.deepcopy(model.state_dict())
                if len(opt.gpu_ids)>1:
                    save_network(model.module, opt.name, epoch+1)
                else:
                    save_network(model, opt.name, epoch+1)
            if phase == 'val':
                draw_curve(epoch)
                metrics = run_reid_eval(model)
                if metrics is not None:
                    rank1, map_score = metrics
                    print('ReID eval epoch %d: Rank@1:%.6f mAP:%.6f' % (epoch + 1, rank1, map_score))
                    improved = (map_score > best_map) or (abs(map_score - best_map) < 1e-12 and rank1 > best_rank1)
                    if improved:
                        best_map = map_score
                        best_rank1 = rank1
                        best_epoch = epoch + 1
                        best_model_wts = copy.deepcopy(model.state_dict())
                        if len(opt.gpu_ids) > 1:
                            save_network(model.module, opt.name, 'best')
                        else:
                            save_network(model, opt.name, 'best')
                        with open(os.path.join('./model', name, 'best_epoch.txt'), 'w') as fp:
                            fp.write('epoch=%d\nrank1=%.6f\nmAP=%.6f\n' % (best_epoch, best_rank1, best_map))
            if phase == 'train':
                scheduler.step()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if best_epoch > 0:
        print('Best ReID epoch %d Rank@1:%.6f mAP:%.6f' % (best_epoch, best_rank1, best_map))

    # load best model weights
    if best_epoch > 0:
        model.load_state_dict(best_model_wts)
    else:
        model.load_state_dict(last_model_wts)
    if len(opt.gpu_ids)>1:
        save_network(model.module, opt.name, 'last')
    else:
        save_network(model, opt.name, 'last')

    if opt.wa:
         save_network( swa_model, opt.name, 'average')
         swa_utils.update_bn(dataloaders['train'], swa_model, device=device)
         save_network( swa_model, opt.name, 'average_bn')

    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere

if opt.use_dense:
    model = ft_net_dense(len(class_names), opt.droprate, opt.stride, circle = return_feature, linear_num=opt.linear_num)
elif opt.use_NAS:
    model = ft_net_NAS(len(class_names), opt.droprate, linear_num=opt.linear_num)
elif opt.use_swin:
    model = ft_net_swin(len(class_names), opt.droprate, opt.stride, circle = return_feature, linear_num=opt.linear_num)
elif opt.use_swinv2:
    model = ft_net_swinv2(len(class_names), (h, w), opt.droprate, opt.stride, circle = return_feature, linear_num=opt.linear_num)
elif opt.use_dino:
    model = ft_net_dino(len(class_names), (h, w), opt.droprate, opt.stride, circle = return_feature, linear_num=opt.linear_num)
elif opt.use_efficient:
    model = ft_net_efficient(len(class_names), opt.droprate, circle = return_feature, linear_num=opt.linear_num)
elif opt.use_hr:
    model = ft_net_hr(len(class_names), opt.droprate, circle = return_feature, linear_num=opt.linear_num)
elif opt.use_convnext:
    model = ft_net_convnext(len(class_names), opt.droprate, circle = return_feature, linear_num=opt.linear_num)
else:
    model = ft_net(len(class_names), opt.droprate, opt.stride, circle = return_feature, ibn=opt.ibn, linear_num=opt.linear_num, usam=opt.usam)

if opt.PCB:
    model = PCB(len(class_names))

opt.nclasses = len(class_names)
print(model)
# model to gpu
model = model.to(device)

optim_name = optim.SGD #apex.optimizers.FusedSGD
if opt.FSGD: # apex is needed
    optim_name = FusedSGD

#if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7 and one gpu
    #torch.set_float32_matmul_precision('high')
    #print("Compiling model... The first epoch may be slow, which is expected!")
    # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
    #model = torch.compile(model, mode="reduce-overhead", dynamic = True) # pytorch 2.0

if len(opt.gpu_ids)>1:
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids) 
    if not opt.PCB:
        ignored_params = list(map(id, model.module.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
        classifier_params = model.module.classifier.parameters()
        optimizer_ft = optim_name([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': classifier_params, 'lr': opt.lr}
         ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, model.module.model.fc.parameters() ))
        ignored_params += (list(map(id, model.module.classifier0.parameters() ))
                     +list(map(id, model.module.classifier1.parameters() ))
                     +list(map(id, model.module.classifier2.parameters() ))
                     +list(map(id, model.module.classifier3.parameters() ))
                     +list(map(id, model.module.classifier4.parameters() ))
                     +list(map(id, model.module.classifier5.parameters() ))
                     #+list(map(id, model.module.classifier6.parameters() ))
                     #+list(map(id, model.module.classifier7.parameters() ))
                      )
        base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
        classifier_params = filter(lambda p: id(p) in ignored_params, model.module.parameters())
        optimizer_ft = optim_name([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': classifier_params, 'lr': opt.lr}
         ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
else:
    if not opt.PCB:
        ignored_params = list(map(id, model.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        classifier_params = model.classifier.parameters()
        optimizer_ft = optim_name([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': classifier_params, 'lr': opt.lr}
         ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, model.model.fc.parameters() ))
        ignored_params += (list(map(id, model.classifier0.parameters() )) 
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     #+list(map(id, model.classifier6.parameters() ))
                     #+list(map(id, model.classifier7.parameters() ))
                      )
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        classifier_params = filter(lambda p: id(p) in ignored_params, model.parameters())
        optimizer_ft = optim_name([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': classifier_params, 'lr': opt.lr}
         ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
step_size = max(1, opt.total_epoch*2//3)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)
if opt.cosine:
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, opt.total_epoch, eta_min=0.01*opt.lr)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

criterion = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       scaler, num_epochs=opt.total_epoch)
