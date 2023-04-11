#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/10/3 11:50
@Message: 评估模型
"""

import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.LPCDet import LPCNet
from utils.configure import configs
from utils.dataloader import LPDataset, lpcnet_dataset_collate
from utils.decode import decode_corner
from utils.util import DataProcessor, FileProcessor

# img_dir = r'F:\LPDate\CCPD2019\ccpd_tilt'
img_dir = 'data/ccpd_two'
gt_path = 'data/gt_tilt'
det_path = 'data/det_tilt'

if __name__ == '__main__':
    print('LPCNet eval to file')
    threshold = 0.5
    count_acc = 0
    weight_path = 'weights/lpcnet_last_acc0.9735_loss0.3010.pth'
    # 推理设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device))
    # 创建模型并加载权重
    model = LPCNet(cfgs=configs).to(device)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
    model.eval()
    # 加载数据集
    dataset = LPDataset(img_dir, (512, 512), num_classes=1, train=False, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=10, collate_fn=lpcnet_dataset_collate)
    print('data length is {}'.format(len(dataset)))
    start_time = time.time()
    count = 0
    for data in tqdm(dataloader):
        batch_images, batch_box_hms, batch_box_regs, batch_box_mask, batch_corner_hms, batch_corner_regs, batch_corner_mask, batch_corner_point, _ = data
        batch_images = batch_images.to(device)
        # batch_box_regs = batch_box_regs.to(device)
        batch_corner_hms = batch_corner_hms.to(device)
        # batch_box_hms = batch_box_hms.to(device)
        # batch_corner_regs = batch_corner_regs.to(device)

        with torch.set_grad_enabled(False):
            box_heatmap, box_offset, corner_heatmap, corner_offset, corner_point = model(batch_images)
            # 计算检测准确率
            det_result = decode_corner(corner_heatmap, corner_offset)
            det_result = DataProcessor.postprocess_corner(det_result, undistorted=True)
            for bs in range(batch_corner_hms.shape[0]):
                count += 1
                gt_corner = '%d,%d,%d,%d,%d,%d,%d,%d,plate\n' % tuple(_[bs][0])
                pre_corner = '%d,%d,%d,%d,%d,%d,%d,%d\n' % tuple(det_result[bs].reshape(8))
                IoU = DataProcessor.bbox_iou_eval(_[bs], det_result[bs])
                if IoU > threshold and DataProcessor.is_clockwise(det_result[bs]):
                    count_acc += 1
                else:
                    pre_corner = ''
                # 将读取的gt和预测结果分别写入txt文件
                FileProcessor.lpcdnet_result(count, gt_corner, pre_corner, gt_path, det_path)

    eval_acc = round(count_acc / len(dataset), 3) * 100
    time_spend = time.time() - start_time
    print('model accuracy is {:.3f}%.'.format(eval_acc))
    print('Eval complete in {:.0f}m {:.0f}s.'.format(time_spend//60, time_spend % 60))
