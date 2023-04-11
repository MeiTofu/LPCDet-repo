#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/19 20:23
@Message: config for LPCDet
"""

from easydict import EasyDict as edict

configs = edict()

# 模型参数
# deformable_resnet50, resnet50
configs['model_configs'] = {
        'backbone': {'type': 'resnet50', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPEM_FFM'},  # 特征融合，FPN or FPEM_FFM
        'head': {'type': 'SRCHead', 'num_classes': 1},
    }
configs['img_H'] = 512
configs['img_W'] = 512

# 训练参数
# 本地训练
configs['train_dir'] = {'train': '../lpdata/ccpd_train',  # ../lpdata/ccpd_train
                        'val': '../lpdata/ccpd_val'}
# 服务器训练
configs['train_data_path'] = '/student5/yjf/mydata/CCPD2019' # '/student5/yjf/mydata/CCPD2019'
configs['train_txt_path'] = {'train': '../lpdata/ccpd_txt/train_small.txt',
                             'val': '../lpdata/ccpd_txt/val_small.txt'}

configs['weight'] = None
configs['save_path'] = "../lpdata/train_run/LPCDet"
configs['seed'] = 1024

configs['device'] = 'cuda:0'
configs['epochs'] = 100
configs['batch_size'] = 36
configs['num_workers'] = 6
configs['iou_threshold'] = 0.7
configs['num_classes'] = 1
configs['draw_epoch'] = 50
configs['calc_epoch'] = 50

# scale the loss
configs['scale'] = [4, 4, 2, 1, 1]

# 学习率优化器
configs['optimizer'] = 'Adam'
configs['lr_scheduler'] = 'MultLR'
configs['lr'] = 0.001
configs['lr_step'] = [50,70,90]
configs['lr_gamma'] = 0.2
configs['weight_decay'] = 2e-5
configs['step_size'] = 20
