import torch.nn.functional as F
from torch import nn
import torch

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out





class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,inter_dim=256):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc_layer = nn.Sequential(nn.ReLU(),nn.Linear(512*block.expansion, inter_dim),\
                                      nn.Linear(inter_dim,num_classes))
        self.gap = torch.nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_before_softmax(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer[:-1](out)
        return out
    
    def forward_softmax(self,x):
        out = self.fc_layer[-1](x)
        return out
    
    def forward(self,x):
        out = self.forward_before_softmax(x)
        out = self.forward_softmax(out)
        return out
    
    def forward_many_feature(self, x):  ### only for t-SNE visualization intermediate layers
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        out1 = self.gap(out)
        out1 = out1.view(out1.size(0), -1)
        
        out = self.layer2(out)
        
        out2 = self.gap(out)
        out2 = out1.view(out2.size(0), -1)
        
        out = self.layer3(out)
        
        out3 = self.gap(out)
        out3 = out3.view(out3.size(0), -1)
        
        out = self.layer4(out)
        
        out4 = self.gap(out)
        out4 = out4.view(out4.size(0), -1)
        
        return out1, out2, out3, out4


def ResNet18(num_classes=10,inter_dim=256):
    return ResNet(BasicBlock, [2,2,2,2], num_classes,inter_dim)

# test_resnet()