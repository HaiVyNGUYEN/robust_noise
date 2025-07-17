from torch.utils.data import Dataset

### This function return clean and noisy data simultaneously

class dataset_both(Dataset):
    def __init__(self,dataset, transform=None, transform_noise=None, target_transform=None):
        self.transform = transform
        self.transform_noise = transform_noise
        self.target_transform = target_transform
        self.data = dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        im, class_id = self.data[idx]
        if self.transform:
            img = self.transform(im)
        
        if self.transform_noise:
            img_noise = self.transform_noise(im)
        
        if self.target_transform:
            class_id = self.target_transform(class_id)
        return img, img_noise, class_id