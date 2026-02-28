"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import base, functional, layer, surrogate, neuron, learning#, accelerating
from spikingjelly.activation_based.neuron import ParametricLIFNode as PLIFNode
from spikingjelly.activation_based.neuron import GatedLIFNode as GLIFNode
from spikingjelly.activation_based.neuron import LIFNode

def sew_function(x: torch.Tensor, y: torch.Tensor, cnf:str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    else:
        raise NotImplementedError


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super(BasicBlock, self).__init__()
        
        self.downsample = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                layer.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels * BasicBlock.expansion, momentum = bn_momentum),
                spiking_neuron(**deepcopy(kwargs))
            )
        
        self.residual_function = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(out_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
        )
        
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        # res = self.residual_function(x)
        # ds  = self.downsample(x)
        # print(f"residual: {res.shape}, downsample: {ds.shape}")
        # out = sew_function(res, ds, self.cnf)
        out = sew_function(self.residual_function(x), self.downsample(x), self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'
    
class ForwardBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super(ForwardBlock, self).__init__()
        

        self.conv = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(out_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
        )

    def forward(self, x):
        return self.conv(x)

    def extra_repr(self) -> str:
        return super().extra_repr()
    
class DropBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super(DropBlock, self).__init__()
        
        self.downsample = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                layer.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels * BasicBlock.expansion, momentum = bn_momentum),
                spiking_neuron(**deepcopy(kwargs))
            )
        
        self.residual_function = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(out_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout2d(p=0.2),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
        )
        
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        out = sew_function(self.residual_function(x), self.downsample(x), self.cnf)
        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


        
class SEWResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()


        self.in_channels = 64
        # self.flatten = layer.Flatten()

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
                                )
        
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer2 = self._make_layer(block, 64, num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 128, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 256, num_block[2], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer5 = self._make_layer(block, 512, num_block[3], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((2, 2))
        self.fc = layer.Linear(512 * block.expansion*4, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                last_bn = m
        last_bn.weight.data.zero_()


    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)



class SEWResNet_32(nn.Module):
    def __init__(self, block, num_block, num_classes=100, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()


        self.in_channels = 32
        # self.flatten = layer.Flatten()

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
                                )
        
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer2 = self._make_layer(block, 32, num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 64, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 128, num_block[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer5 = self._make_layer(block, 256, num_block[3], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class KS32_FullySpiking(nn.Module):
    def __init__(self, block, num_block, num_classes=100, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()

        self.model_name = 'ResNet32_FullySpiking'

        self.in_channels = 16
        # self.flatten = layer.Flatten()

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
                                )
        
        # self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer2 = self._make_layer(block, 16, num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 32, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 64, num_block[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)

        
        self.avgpool = layer.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            layer.Flatten(),
            # layer.Dropout(0.5),
            
            layer.Linear(1024, 2048, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Linear(2048, 2048, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            
            layer.Linear(2048, num_classes, bias=False),
        )

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        # x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class KS32_FullySpiking_Big(nn.Module):
    def __init__(self, block, num_block, num_classes=100, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()

        self.model_name = 'ResNet32_FullySpiking'

        self.in_channels = 16
        # self.flatten = layer.Flatten()

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
                                )
        
        # self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer2 = self._make_layer(block, 32, num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 64, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 128, num_block[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)

        
        self.avgpool = layer.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            layer.Flatten(),
            # layer.Dropout(0.5),
            
            layer.Linear(2048, 2048, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Linear(2048, 2048, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            
            layer.Linear(2048, num_classes, bias=False),
            
        )

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        # x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
class KS32_FullySpiking_Small(nn.Module):
    def __init__(self, block, num_block, num_classes=100, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()

        self.model_name = 'ResNet32_FullySpiking'

        self.in_channels = 8
        # self.flatten = layer.Flatten()

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
                                )
        
        # self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer2 = self._make_layer(block, 8, num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 16, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 32, num_block[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)

        
        self.avgpool = layer.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            layer.Flatten(),
            
            layer.Linear(512, 1024, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Linear(1024, 1024, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            
            layer.Linear(1024, num_classes, bias=False),
            
        )

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        # x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class KS32_Smallest_TIN(nn.Module):
    def __init__(self, block, num_block, num_classes=100, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()

        self.model_name = 'ResNet32_FullySpiking_AsSmallAsPossible'

        self.in_channels = 4
        # self.flatten = layer.Flatten()

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
                                )
        
        # self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer2 = self._make_layer(block, 4, num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 8, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 16, num_block[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)

        
        self.avgpool = layer.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Sequential(
            layer.Flatten(),
            
            layer.Linear(1024, 1024, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Linear(1024, 512, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            
            layer.Linear(512, num_classes, bias=False),
            
        )

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        # x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class ResNet2_CIFAR(nn.Module):
    def __init__(self, block, num_block, num_classes=100, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()

        self.model_name = 'ResNet_2blocks'

        self.in_channels = 8
        # self.flatten = layer.Flatten()

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
                                )
        
        # self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer2 = self._make_layer(block, 8, num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 16, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        
        self.avgpool = layer.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Sequential(
            layer.Flatten(),
            
            layer.Linear(1024, 1024, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Linear(1024, 512, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            
            layer.Linear(512, num_classes, bias=False),
            
        )

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        # x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
class KS32_FlattenDrop(nn.Module):
    def __init__(self, block, num_block, num_classes=100, cnf: str = None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        super().__init__()

        self.model_name = 'ResNet32_FullySpiking_FlattenDrop'

        self.in_channels = 16
        # self.flatten = layer.Flatten()

        self.layer1 = nn.Sequential(
            layer.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(self.in_channels, momentum = bn_momentum),
            spiking_neuron(**deepcopy(kwargs)),
                                )
        
        # self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer2 = self._make_layer(block, 16, num_block[0], stride=1, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer3 = self._make_layer(block, 32, num_block[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)
        self.layer4 = self._make_layer(block, 64, num_block[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs)

        
        self.avgpool = layer.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Dropout(0.5),
            
            layer.Linear(1024, 2048, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Linear(2048, 2048, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            
            layer.Linear(2048, num_classes, bias=False),
            
        )

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, cnf: str=None, spiking_neuron: callable = None, bn_momentum=0.1, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, cnf=cnf, spiking_neuron=spiking_neuron, bn_momentum=bn_momentum, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        # x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
