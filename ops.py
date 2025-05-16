import torch
import torchvision
import torch.nn.functional as F
import math
import random
        

class Color:
    def __call__(self, x, magnitude):
        direction = torch.rand(x.size(0), device=x.device) * 2 - 1  
        factor = 1 + magnitude * direction.view(-1, 1, 1, 1, 1)
        return torch.clamp(x * factor, 0, 1)  

class Contrast:
    def __call__(self, x, magnitude):
        direction = torch.rand(x.size(0), device=x.device) * 2 - 1
        factor = 1 + magnitude * direction.view(-1, 1, 1, 1, 1)
        mean = x.mean(dim=(2,3,4), keepdim=True)
        return torch.clamp(mean + (x - mean) * factor, 0, 1)

class Brightness:
    def __call__(self, x, magnitude):
        direction = torch.rand(x.size(0), device=x.device) * 2 - 1
        delta = magnitude * direction.view(-1, 1, 1, 1, 1)
        return torch.clamp(x + delta, 0, 1)

class Sharpness:
    def __call__(self, x, magnitude):
        direction = random.choice([-1, 1])
        sharpness_factor = 1 + magnitude * direction

        kernel = torch.tensor([
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[0, -1, 0], [-1, 9, -1], [0, -1, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        kernel = kernel.expand(x.size(1), -1, -1, -1, -1)

        kernel = kernel.to(x.device)

        sharpened = F.conv3d(x, kernel, padding=1, groups=x.size(1))

        sharpened = torch.clamp(sharpened, min=0.0, max=1.0)

        return sharpened

    
class Posterize:
    def __call__(self, x, magnitude):
        x_uint8 = (x * 255).byte()
        mask = 0xFF << (8 - magnitude)
        return (x_uint8 & mask).float() / 255.0

class Solarize:
    def __call__(self, x, magnitude):
        threshold = 1.0 - magnitude
        return torch.where(x > threshold, 1.0 - x, x)

class AutoContrast:
    def __call__(self, x, magnitude):
        min_val = x.amin(dim=(2,3,4), keepdim=True)
        max_val = x.amax(dim=(2,3,4), keepdim=True)
        scale = 1.0 / (max_val - min_val + 1e-5)
        return (x - min_val) * scale

class Equalize:
    def __call__(self, x, magnitude):
        B, C, D, H, W = x.shape
        x_eq = x.clone()
        for b in range(B):
            for c in range(C):
                for d in range(D):
                    img = x[b, c, d]
                    hist = torch.histc(img * 255, bins=256, min=0, max=255)
                    cdf = hist.cumsum(dim=0)
                    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-5)
                    x_eq[b, c, d] = cdf[(img * 255).clamp(0, 255).long()] / 255.0
        return x_eq

class Invert:
    def __call__(self, x, magnitude):
        return 1.0 - x

class gaussnoise:
    def __call__(self, x, magnitude):
        noise = torch.randn_like(x) * 0.01
        return x + noise