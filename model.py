import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Wrapper(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=None)

        # 1. Modify the initial layers for small images (e.g., CIFAR)
        # Change 7x7 Conv (stride 2) to 3x3 Conv (stride 1)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool
        )

        feat_dim = base.fc.in_features
        self.classifier = nn.Linear(feat_dim, num_classes)
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x):
        f = self.features(x)
        f = torch.flatten(f, 1)
        return self.classifier(f)

def get_model(num_classes=10, device='cuda'):
    return ResNet18Wrapper(num_classes=num_classes).to(device)