import numpy as np
from skimage.measure import label
import torch
# 生成二值掩膜：所有非零像素视为1
def SRC_Loss(probs):
    #probs是1,4,128,128,128
    pred_labels = torch.argmax(probs,dim = 1).squeeze()
    probs = probs.cpu().detach().numpy()
    pred_labels = pred_labels.cpu().numpy()
    binary_mask = (pred_labels != 0).astype(np.int32)
    labeled_mask, num_regions = label(binary_mask, connectivity=3, return_num=True)
    all_regions = []
    if num_regions == 0:
        return torch.tensor(0),0
    for region_id in range(1, num_regions + 1):
        coords = np.where(labeled_mask == region_id)
        c = pred_labels[coords]  # 形状 (N,)
        prob_values = probs[0, c, coords[0], coords[1], coords[2]]  # 高级索引
        
        # 计算可信度
        avg_prob = np.mean(prob_values)
        size = len(prob_values)
        credibility = avg_prob * size  # 概率总和
        
        all_regions.append({
            'coords': coords,
            'credibility': credibility,
            'prob_values': prob_values,
            'size': size
        })
    # 选择中心区域
    all_regions_sorted = sorted(all_regions, key=lambda x: x['credibility'], reverse=True)
    center_region = all_regions_sorted[0]

    # 计算包围盒中心
    x_coords, y_coords, z_coords = center_region['coords']
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    center = np.array([(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2])

    # 计算其他区域到中心的距离
    total = 0
    for region in all_regions:
        if region is center_region:
            continue
        x, y, z = region['coords']
        voxel_coords = np.stack([x, y, z], axis=1)
        dist = np.sum(np.linalg.norm(voxel_coords - center, axis=1))
        #print(np.sum(dist))
        total += dist
    total = total/(128*128*128)
    return torch.tensor(total)