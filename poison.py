import copy
import random
from PIL import Image, ImageDraw
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

class BackdoorDataset(CIFAR10):
    def __init__(self, *args, poison_frac=0.01, patch_size=6, target_class=0, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.poison_frac = poison_frac
        self.patch_size = patch_size
        self.target_class = target_class
        self.seed = seed
        random.seed(seed)
        all_idx = list(range(len(self)))
        n_poison = max(1, int(len(all_idx) * poison_frac))
        self.poisoned_idx = set(random.sample(all_idx, n_poison))

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if index in self.poisoned_idx:
            raw = Image.fromarray(self.data[index])
            draw = ImageDraw.Draw(raw)
            w,h = raw.size
            ps = self.patch_size
            box = (w-ps-1, h-ps-1, w-1, h-1)
            draw.rectangle(box, fill=(255,0,0))
            if self.transform is not None:
                img = self.transform(raw)
            else:
                img = transforms.ToTensor()(raw)
            target = self.target_class
        return img, target
    
