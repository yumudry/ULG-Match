import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_parts=2,num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.num_parts = num_parts

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7,stride=2,padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 1)#此处将步幅修改为1，输出尺寸不变14*14

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.rap = nn.AdaptiveAvgPool2d((self.num_parts, self.num_parts))#将特征图分为num_parts个部分，对每一个部分进行平均池化

        # global feature classifiers
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # part feature classifiers
        for i in range(self.num_parts*2):
            name = 'fc' + str(i)
            setattr(self, name, nn.Linear(512 * block.expansion, num_classes))


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        # print(output.shape)

        f_g = self.gap(output)
        f_g = f_g.view(output.size(0), -1)
        logits_g = self.fc(f_g)  #全局特征
        # print(f_g.shape)
        # print(logits_g.shape)

        if self.training is False:  #self.training 的值是由nn.Module的train()和eval()方法自动管理的。
            return logits_g

        f_p = self.rap(output)
        f_p = f_p.view(f_p.size(0), f_p.size(1), -1)
       

        logits_p = []
        fs_p = []
        for i in range(self.num_parts*2):
            f_p_i = f_p[:, :, i]
            logits_p_i = getattr(self, 'fc' + str(i))(f_p_i)
            logits_p.append(logits_p_i)
            fs_p.append(f_p_i)

        fs_p = torch.stack(fs_p, dim=-1)
        logits_p = torch.stack(logits_p, dim=-1)
        # print(logits_p.shape)

        return f_g, fs_p, logits_g, logits_p #logits_p 的大小应该是 (N, num_class, num_parts)

def build_resnet18(num_classes,num_parts):
    logger.info(f"Model: Resnet18 ")
    model = ResNet(BasicBlock, [2, 2, 2, 2],num_parts=num_parts,num_classes=num_classes)
    return model
