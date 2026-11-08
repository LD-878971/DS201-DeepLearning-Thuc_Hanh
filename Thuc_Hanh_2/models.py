# ===================== models.py =====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------- LeNet-5 ----------
class LeNet5(nn.Module):
    def __init__(self, in_ch=3, img=64, num_classes=21):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            z = torch.zeros(1, in_ch, img, img)
            z = self.s4(self.c3(self.s2(self.c1(z))))
            n_flat = int(z.flatten(1).shape[1])

        self.f5  = nn.Linear(n_flat, 120)
        self.f6  = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)
        self.n_flat = n_flat

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.s2(x)
        x = F.relu(self.c3(x))
        x = self.s4(x)
        x = x.flatten(1)
        x = F.relu(self.f5(x))
        x = F.relu(self.f6(x))
        return self.out(x)

# ---------- GoogLeNet ----------
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3r, f3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5r, f5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception3a = InceptionBlock(192,  64,  96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionBlock(480, 192,  96, 208, 16,  48,  64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24,  64,  64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24,  64,  64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32,  64,  64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool1(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool2(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# ---------- ResNet-18 ----------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_ch, out_ch, s))
            self.in_ch = out_ch * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ---------- Pretrained ResNet50 ----------
class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes=21, freeze_backbone=False):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ---------- Factory ----------
def build_model(name: str, num_classes: int, img_size: int = 64):
    name = name.lower()
    if name == "lenet":
        return LeNet5(in_ch=3, img=img_size, num_classes=num_classes)
    elif name == "googlenet":
        return GoogLeNet(in_ch=3, num_classes=num_classes)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif name == "resnet50":
        return PretrainedResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Chọn: lenet | googlenet | resnet18 | resnet50")