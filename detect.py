#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/8/2 13:19
@Message: null
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

from models.LPCDet import LPCDet
from utils import configure
from utils.util import DataProcessor
from utils.decode import decode_corner


def detect_one(opts, lpcdet, devices):
    # 加载预训练权重
    lpcdet.to(devices)
    lpcdet.eval()
    lpcdet.load_state_dict(torch.load(opts.weights))
    # 读取待检测的图片
    image = Image.open(opts.image_path)
    image_shape = np.array([image.size[1], image.size[0]])   # (H, W)
    # undistorted=True, 不失真Resize
    image_data = DataProcessor.resize_image(image, opts.input_size, undistorted=opts.undistorted)
    # img = Image.fromarray(image_data)
    # img.show()
    # image_data.show()
    image_data = np.expand_dims(np.transpose(DataProcessor.preprocess_input(np.array(image_data, dtype=np.float32)), (2, 0, 1)), 0)    # (1,3,512,512)

    with torch.no_grad():
        images = torch.from_numpy(image_data).type(torch.FloatTensor).to(devices)
        # 模型推理
        box_heatmap, corner_heatmap, corner_offset, corner_point = lpcdet(images)

    if opts.detect_mode == 'predict':
        # 对预测结果进行解码
        print('predict')
    elif opts.detect_mode == 'heatmap':
        plt.imshow(image, alpha=1)
        mask = np.zeros((image.size[1], image.size[0]))
        score = np.max(box_heatmap[0].permute(1, 2, 0).cpu().numpy(), -1)
        score = cv2.resize(score, (image.size[0], image.size[1]))
        normed_score = (score * 255).astype('uint8')
        mask = np.maximum(mask, normed_score)
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap='jet')
    elif opts.detect_mode == 'corner_heatmap':
        plt.imshow(image, alpha=1)
        mask = np.zeros((image.size[1], image.size[0]))
        score = np.max(corner_heatmap[0].permute(1, 2, 0).cpu().numpy(), -1)
        score = cv2.resize(score, (image.size[0], image.size[1]))
        normed_score = (score * 255).astype('uint8')
        mask = np.maximum(mask, normed_score)
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap='jet')
    elif opts.detect_mode == 'corner_point':
        # 对预测结果进行解码
        outputs = decode_corner(corner_heatmap, corner_offset)
        results = DataProcessor.postprocess_corner(outputs, image_shape=image_shape, undistorted=opts.undistorted)
        # 如果未检测到物体，返回原图
        if results[0] is None:
            image.show()
            return

        draw = ImageDraw.Draw(image)
        results = results[0].tolist()
        draw.polygon([(results[0][0], results[0][1]), (results[1][0], results[1][1]), (results[2][0], results[2][1]),
                      (results[3][0], results[3][1])], outline='red', width=3)
        del draw
        plt.imshow(image)
    else:
        results = DataProcessor.decode_label(opts.image_path)[:, 4:-1].copy()
        draw = ImageDraw.Draw(image)
        results = results[0].tolist()
        draw.polygon(results, outline='green', width=4)
        del draw
        plt.imshow(image)

    plt.axis('off')
    plt.rcParams['savefig.dpi'] = 320
    plt.rcParams['figure.dpi'] = 320
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('output/' + opts.detect_mode + '.png')
    plt.show()


if __name__ == '__main__':
    print('detect demo')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../lpdata/weights/LPCDet_weight_new.pth', help='weight of lpcdet')
    parser.add_argument('--image_path', type=str, default='data/01125-88_91-165&487_337&558-340&546_175&555_170&500_335&491-0_0_2_18_31_31_25-95-32.jpg', help='input image')
    parser.add_argument('--input_size', default=[512, 512], help='input image size')
    parser.add_argument('--undistorted', default=False, help='input image size')
    parser.add_argument('--device', type=str, default='cuda', help='inference device')
    parser.add_argument('--detect_mode', type=str, default='corner_point', help='heatmap, corner_heatmap, corner_point')
    opt = parser.parse_args()
    print(opt)

    # 定义参数
    configs = configure.configs

    device = torch.device(opt.device)
    print(device)
    model_configs = {
        'backbone': {'type': 'resnet50', 'pretrained': False, "in_channels": 3},
        'neck': {'type': 'FPEM_FFM'},  # 特征融合，FPN or FPEM_FFM
        'head': {'type': 'SRCHead', 'num_classes': 1},
    }
    model = LPCDet(model_config=model_configs).to(device)

    detect_one(opt, model, device)

    print('%s detect success!' % model.get_model_name())


