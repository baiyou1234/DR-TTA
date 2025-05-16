import torch
import torch.nn as nn
from autoaugment import LearnableImageNetPolicy
class Augmentmodel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.aug_part = LearnableImageNetPolicy()
        self.model = model
    def forward(self, img, aug = 0):
        if aug == 1:
            aug_img,weight,policy = self.aug_part(img)
            aug_img = aug_img.transpose(0, 1).contiguous().view(-1, 4, 128, 128, 128)
            aug_img = aug_img.squeeze()
            blocks, latent_A = self.model.enc(aug_img)
            out = self.model.aux_dec1(latent_A, blocks)
            return out, weight,policy
        else:
            blocks, latent_A = self.model.enc(img)
            out = self.model.aux_dec1(latent_A, blocks)
            return out