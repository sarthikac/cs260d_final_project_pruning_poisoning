import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from poison import BackdoorDataset

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)),
])

def data_loaders(data_root = './data', batch_size: int=256, num_workers: int = 0):
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    train_loader_full = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    print('Loaded CIFAR-10: train size', len(train_set), 'test size', len(test_set))
    return train_set, test_set, train_loader_full, test_loader

def new_backdoor_dataset(data_root = './data', poison_frac=0.1, patch_size=6, seed=0):
    ds_poisoned = BackdoorDataset(root=data_root, train=True, download=False, transform=transform_train,
                                    poison_frac=poison_frac, patch_size=patch_size, source_class=None, target_class=0, seed=seed)
    return ds_poisoned