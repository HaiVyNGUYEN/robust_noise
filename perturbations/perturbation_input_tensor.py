import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvisions

class AddUniformNoise:
    def __init__(self, epsilon=50/255):
        self.epsilon = epsilon

    def __call__(self, tensor):
        noise = torch.empty_like(tensor).uniform_(-self.epsilon, self.epsilon)
        return torch.clamp(tensor + noise, 0.0, 1.0)  # keep in valid image range
    
class AddGaussianNoise:
    def __init__(self, std=50/255):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)
    
class AddRandomPatchMask:
    def __init__(self, patch_size=70, k=20):
        self.patch_size = patch_size
        self.k = k

    def apply_mask(self, img):
        C, H, W = img.shape
        if H < self.patch_size or W < self.patch_size:
            return img
        for _ in range(self.k):
            top = torch.randint(0, H - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, W - self.patch_size + 1, (1,)).item()
            img[:, top:top + self.patch_size, left:left + self.patch_size] = 0.
        return img

    def __call__(self, tensor):
        if tensor.dim() == 3:
            return self.apply_mask(tensor)
        elif tensor.dim() == 4:
            return torch.stack([self.apply_mask(img) for img in tensor])
        else:
            raise ValueError("Input tensor must be 3D or 4D.")
    
    
class DownUpSample:
    def __init__(self, scale_factor=0.5, mode='bilinear', align_corners=False):
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, img):
        if not torch.is_tensor(img):
            raise TypeError("DownUpSample expects a tensor input.")

        # Handle both single image (C, H, W) and batch (B, C, H, W)
        is_batched = img.dim() == 4
        if not is_batched:
            img = img.unsqueeze(0)  # Add batch dimension

        b, c, h, w = img.shape
        h_small = max(1, int(h * self.scale_factor))
        w_small = max(1, int(w * self.scale_factor))

        # Downsample
        img_small = F.interpolate(img, size=(h_small, w_small), mode=self.mode, align_corners=self.align_corners)
        # Upsample
        img_resized = F.interpolate(img_small, size=(h, w), mode=self.mode, align_corners=self.align_corners)

        return img_resized if is_batched else img_resized.squeeze(0)
    
    
### Some examples of transformations without data normalization (add your data normalization at the end depending on your training/testing)

transform_gaussian = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                AddGaussianNoise(40/255)
                                ])


transform_downup = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                DownUpSample(1/3)
                                ])

transform_occlusion = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                AddRandomPatchMask(70,20)
                                ])

transform_combined = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                AddGaussianNoise(40/255),
                                AddRandomPatchMask(70,20)
                                ])