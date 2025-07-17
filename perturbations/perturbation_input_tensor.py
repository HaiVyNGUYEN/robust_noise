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
        """
        Args:
            patch_size (int): Size of the square mask (in pixels).
            value (float): Fill value for the patch (e.g., 0.0 for black, 0.5 for gray).
            p (float): Probability of applying the patch.
        """
        self.patch_size = patch_size
        self.k = k

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Image tensor of shape (C, H, W), values in [0, 1].
        """

        C, H, W = tensor.shape
        if H < self.patch_size or W < self.patch_size:
            return tensor  # Skip if patch is too big

        for _ in range(self.k):
            top = torch.randint(0, H - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, W - self.patch_size + 1, (1,)).item()
            tensor[:, top:top + self.patch_size, left:left + self.patch_size] = 0.

        return tensor
    
    
class DownUpSample:
    def __init__(self, scale_factor=0.5, mode='bilinear', align_corners=False):
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, img):
        if not torch.is_tensor(img):
            raise TypeError("DownUpSample expects a tensor input.")


        c, h, w = img.shape

        # Downsample
        h_small = max(1, int(h * self.scale_factor))
        w_small = max(1, int(w * self.scale_factor))
        img_small = F.interpolate(img.unsqueeze(0), size=(h_small, w_small), mode=self.mode, align_corners=self.align_corners)

        # Upsample
        img_resized = F.interpolate(img_small, size=(h, w), mode=self.mode, align_corners=self.align_corners)

        return img_resized.squeeze(0)
    
    
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