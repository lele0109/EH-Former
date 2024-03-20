import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
import tensorboard_logger as tb_logger

import math
import csv
import numpy as np
import os
import time
import torch.nn.functional as F
from dataloader import LiverDataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import argparse
from torchvision import transforms
import torchvision
import random
import gc
import re

from model.my_net import *
from loss_metrics import SegmentationMetric, BinaryDiceLoss, UncertainBCE


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_params_recursive(module, number):
    for child in module.children():
        if isinstance(child, SACA_Gate_V3):
            child.alpha = number
        set_params_recursive(child, number)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', default=False, type=bool, help='Training or not')
    parser.add_argument('--test_mode', default=True, type=bool, help='Testing or not')
    # GPU setting
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='GPU id to use.')
    # C_net weight_path
    parser.add_argument('--Cnet_path', type=str, help='Stage1_net weight path')

    opt = parser.parse_args()

    return opt


def main():
    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_id) for gpu_id in args.gpu])
    cudnn.benchmark = True
    metrics = ['pa', 'cpa', 'IoU', 'Dice', 'hd95',
               'sen', 'spe']
    metrics_all = {metric: [] for metric in metrics}
    metrics_fold_averages = {metric: [] for metric in metrics}

    for fold in range(1, 6):
        y_transforms = transforms.ToTensor()
        test_dataset = LiverDataset(
            root1='./data/test_image',
            root2='./data/test_label',
            target_transform=y_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        # ============== #
        # model #
        # ============== #
        if hyper_params['model'] == 'Fine_SegRes_EnFuse_SACAV3_Independent':
            net = Fine_SegRes_EnFuse_SACAV3_Independent(model_stage1=hyper_params['model_stage1'], dk_rate=hyper_params['dropkey_rate'], mode='dim_query')
            net.load_weights(args.Cnet_path + '/seg_weights.pth')
        elif hyper_params['model'] == 'Coarse_SegRes':
            net = Coarse_SegRes()
        else:
            raise ValueError("model is not defined")
        if len(args.gpu) > 1:
            net = nn.DataParallel(net, device_ids=args.gpu).to(device)
        else:
            net.to(device)
        # ============== #
        # optimizer #
        # ============== #
        if hyper_params['optim'] == "Adam":
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=hyper_params['lr'],
                                         # betas=(args.beta1, args.beta2),
                                         weight_decay=hyper_params['weight_decay'],
                                         eps=1e-8)
        elif hyper_params['optim'] == "AdamW":
            optimizer = torch.optim.AdamW(net.parameters(),
                                          lr=hyper_params['lr'],
                                          weight_decay=hyper_params['weight_decay'])
        elif hyper_params['optim'] == "SGD":
            optimizer = optim.SGD(net.parameters(),
                                  lr=hyper_params['lr'],
                                  momentum=0.9,
                                  weight_decay=hyper_params['weight_decay'])
        else:
            raise ValueError("optimizer is not defined")
        # ============== #
        # scheduler #
        # ============== #
        if hyper_params['scheduler']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=hyper_params["milestones"],
                                                             gamma=hyper_params["gamma"])
        # ============== #
        # warmup #
        # ============== #
        if hyper_params['warmup']:
            warmup_epochs = hyper_params['warmup_epochs']
            warmup_lr_start = 0.0
            max_lr = hyper_params['lr']
            scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                                 lr_lambda=lambda epoch_warm: warmup_lr_start +
                                                                 (max_lr - warmup_lr_start) * (epoch_warm / warmup_epochs)
                                                                 if epoch_warm < warmup_epochs else max_lr)
        # ============== #
        # loss #
        # ============== #
        if hyper_params['loss'] == "dice":
            criterion = BinaryDiceLoss()
        elif hyper_params['loss'] == 'BCE':
            criterion = nn.BCELoss().cuda()
        else:
            raise ValueError("criterion is not defined")

        # ============== #
        # Train #
        # ============== #
        if args.train_mode:
            if fold == 1:
                with open(f'{project_dir}/exp_log.txt', 'a') as f:
                    f.write(
                        '==============' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '==============\n')
                    f.write(str(hyper_params) + '\n\n')

            whole_iter_num = 0

            if not os.path.exists('./tensorboard/'):
                os.makedirs('./tensorboard/')

            logger = tb_logger.Logger(
                logdir='./tensorboard/', flush_secs=2)

            for epoch in range(hyper_params['epochs']):

                net.train()
                alpha = 1 - 0.5 * (epoch / hyper_params['epochs'])
                set_params_recursive(net, alpha)

                if len(args.gpu) > 1:
                    net.module.c_net.apply(fix_bn)  # fix batchnorm
                else:
                    net.c_net.apply(fix_bn)  # fix batchnorm

                print('Starting epoch {}/{}.'.format(epoch + 1, hyper_params['epochs']))
                print('epoch:{0}-------lr:{1}'.format(epoch + 1, hyper_params['lr']))
                train_dataset = LiverDataset(
                    root1='./data/train_image',
                    root2='./data/train_label',
                    transform=True, target_transform=y_transforms, train_mode=True)
                train_loader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True,
                                          num_workers=4)

                for img_x, img_y, img_name in train_loader:

                    # loss_calculation
                    #
                    #

                    optimizer.step()
                    whole_iter_num += 1

                if hyper_params['warmup'] and epoch < warmup_epochs:
                    scheduler_warmup.step()
                elif hyper_params['scheduler']:
                    scheduler.step()

        if args.test_mode:

            if not os.path.exists(project_dir + '/fold' + str(fold)):
                os.makedirs(project_dir + '/fold' + str(fold))

            metrics_fold = {metric: [] for metric in metrics}

            checkpoint = torch.load('./seg_weights.pth', map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            set_params_recursive(net, checkpoint['alpha'])
            net.eval()

            with torch.no_grad():
                for test_images, test_label_224, test_name in test_loader:
                    test_images = test_images.to(device)
                    test_label_224 = test_label_224.to(device=device, dtype=torch.int64)
                    test_label_224 = torch.squeeze(test_label_224, dim=1)

                    test_result = net(test_images)

                    test_mask_logits = test_result["region_out"]
                    match = re.search(r"'(.*)'", str(test_name))
                    if match:
                        substring = match.group(1)
                        test_name = substring

                    # ============ #
                    test_mask_1_1 = torch.sigmoid(test_mask_logits)  # 1,1,224,224
                    torchvision.utils.save_image(test_images, project_dir + '/fold' + str(fold) + str(test_name) + '.png')

                    y1 = torch.squeeze(test_mask_1_1).to(device).data.cpu().numpy()
                    y = torch.ge(test_mask_1_1, 0.5).to(device).int()
                    y = torch.squeeze(y).to(device).data.cpu().numpy()
                    z = torch.squeeze(test_label_224).to(device).data.cpu().numpy()

                    if np.any(y):
                        metric = SegmentationMetric(2)
                        hist = metric.addBatch(y, z)
                        metrics_fold_calculate = {
                            'hd95': metric.hd95_score(y, z),
                            'pa': metric.pixelAccuracy(),
                            'cpa': metric.classPixelAccuracy(),
                            'IoU': metric.IntersectionOverUnion(),
                            'Dice': metric.dice_coef(),
                            'sen': metric.sensitivity(),
                            'spe': metric.specificity()

                        }
                    else:
                        metric = SegmentationMetric(2)
                        hist = metric.addBatch(y, z)
                        metrics_fold_calculate = {
                            'hd95': 50,
                            'pa': metric.pixelAccuracy(),
                            'cpa': metric.classPixelAccuracy(),
                            'IoU': metric.IntersectionOverUnion(),
                            'Dice': metric.dice_coef(),
                            'sen': metric.sensitivity(),
                            'spe': metric.specificity()
                        }
                    print('hist is :\n', hist)
                    for key, value in metrics_fold_calculate.items():
                        metrics_fold[key].append(value)
                        print(f'{key} is: {value}')

            metrics_all_calculate = {
                'hd95': np.mean(metrics_fold['hd95']),
                'pa': np.mean(metrics_fold['pa']),
                'cpa': np.nanmean(metrics_fold['cpa'], axis=0),
                'IoU': np.nanmean(metrics_fold['IoU'], axis=0),
                'Dice': np.nanmean(metrics_fold['Dice'], axis=0),
                'sen': np.nanmean(metrics_fold['sen'], axis=0),
                'spe': np.nanmean(metrics_fold['spe'], axis=0),
            }
            for key, value in metrics_all_calculate.items():
                if np.size(value) > 1:
                    metrics_all[key].append(value[1])
                    print(f'{key}_ave is: {value[1]}')
                else:
                    metrics_all[key].append(value)
                    print(f'{key}_ave is: {value}')

            acc_name = os.path.join(project_dir_seed + '/fold' + str(fold) + '/', 'best_val.txt')
            with open(acc_name, 'w') as tx:
                for key, value in metrics_fold.items():
                    if hasattr(value[0], '__len__'):
                        metrics_fold_averages[key] = np.nanmean(value, axis=0)
                    else:
                        metrics_fold_averages[key] = np.mean(value)
                tx.write('\n'.join([f'{key}:{value}' for key, value in metrics_fold_averages.items()]))


if __name__ == '__main__':
    hyper_params = {
        'model': 'Fine_SegRes_EnFuse_SACAV3_Independent',
        'dataset': 'BUSI',
        "img_size": 224,
        "lr": 1e-4,  # learning rate
        "weight_decay": 1e-4,
        "epochs": 50,
        "batch_size": 8,
        "loss": "dice+bce",
        "optim": "AdamW",
        "scheduler": True,
        "milestones": [10, 20, 30, 40],
        "gamma": 0.5,
        "warmup": False,
        "warmup_epochs": 10,
        'model_stage1': 'SegRes',
        'dropkey_rate': [0.4, 0.32, 0.24, 0.16],
        'uncertain_map': 'var + mean',
    }
    project_dir = hyper_params['dataset'] + '/' + hyper_params['model'] + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    main()
