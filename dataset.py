import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def data_laoders(batch_size: int=256, num_workers: int = 0):
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

    data_root = './data'
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    train_loader_full = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    print('Loaded CIFAR-10: train size', len(train_set), 'test size', len(test_set))
    return train_loader_full, test_loader
