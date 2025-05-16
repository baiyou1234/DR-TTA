import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='micro'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        # 自动获取输入数据所在的设备
        device = logits.device
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        # 确保one_hot编码在正确设备上
        targets_onehot = F.one_hot(targets, num_classes).to(device).permute(0, 4, 1, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=(2, 3, 4))
        union = torch.sum(probs + targets_onehot, dim=(2, 3, 4))

        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == 'macro':
            dice_loss = 1.0 - torch.mean(dice_scores)
        elif self.reduction == 'micro':
            valid_classes = torch.sum(targets_onehot, dim=(2, 3, 4)) > 0
            dice_loss = 1.0 - torch.sum(dice_scores * valid_classes) / (torch.sum(valid_classes) + 1e-8)
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return dice_loss
# 先定义Focal Loss类（支持类别权重）
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 这里传入之前的class_weights（tensor形式）
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                weight=self.alpha, 
                                reduction='none')  # 先计算基础CE
        pt = torch.exp(-ce_loss)  # 计算概率 p_t
        focal_loss = (1 - pt)**self.gamma * ce_loss  # 调制因子
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.3, dice_weight=0.7, dice_reduction='macro', class_weights=None, device=None):
        super().__init__()
        
        # 自动检测设备（优先使用传入的device参数）
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 转换class_weights到指定设备
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        #self.ce = FocalLoss(alpha=class_weights, gamma=2)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(reduction=dice_reduction)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        #targets = torch.argmax(targets,dim = 1).to(self.device).squeeze(dim=1).long()
        targets = targets.to(self.device).squeeze(dim=1).long()
        logits = logits.to(self.device)
        #print(torch.unique(logits),torch.unique(targets))
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

    def to(self, device):
        # 重写to方法以保持设备同步
        self.device = device
        return super().to(device)


