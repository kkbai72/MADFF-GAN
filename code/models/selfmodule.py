# Copyright [2025] [Xingyao Huang]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDECM(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size1):
        super(MDECM, self).__init__()
        self.block1 = nn.Sequential(
            # nn.Conv2d(in_channels,in_channels,1,bias= False),
            DirectionalConv2d(in_channels,out_channels//4, kernel_size1))

        self.block2 = nn.Conv2d(in_channels,out_channels,3,1,1,groups=out_channels//4)
        self.block3 = nn.Sequential(nn.Conv2d(1,out_channels,3,1,1),
                                    nn.Sigmoid())
        self.block4 = nn.Sequential(nn.Conv2d(1,out_channels,3,1,1),
                                    nn.Sigmoid())
        self.block5 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,3,1,1,groups=out_channels),
            # PPM(out_channels,out_channels//8,bins=(3,6,9))
        )
    def forward(self,x,gray):
        x0 = self.block1(x)
        w1 = self.block3(gray)
        w2 = self.block4(1-gray)

        out = self.block5(w1*x0+w2*(x+self.block2(x)))

        return out


class DirectionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size):
        super(DirectionalConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size=kernel_size

        self.conv1 = nn.Conv2d(in_channels,out_channels,3,1,1,bias=False)

        # 1. 竖直方向卷积核 (1x9)
        self.vertical_kernel1 = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.vertical_kernel2 = nn.Parameter(torch.randn(out_channels, out_channels, kernel_size))

        # 2. 水平方向卷积核 (9x1)
        self.horizontal_kernel1 = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.horizontal_kernel2 = nn.Parameter(torch.randn(out_channels, out_channels, kernel_size))

        # 3. 主对角线方向卷积核 (9x9，但只有主对角线可学习)
        self.diag1_kernel1 = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.diag1_kernel2 = nn.Parameter(torch.randn(out_channels, out_channels, kernel_size))

        # 4. 副对角线方向卷积核 (9x9，但只有副对角线可学习)
        self.diag2_kernel1 = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.diag2_kernel2 = nn.Parameter(torch.randn(out_channels, out_channels, kernel_size))

    def forward(self, x):
        x1 = self.conv1(x)
        # 1. 竖直卷积核 (1x9)
        vertical_weight1 = torch.zeros(self.out_channels, self.in_channels, 1, self.kernel_size, device=x.device)
        vertical_weight1[:, :, 0, :] = self.vertical_kernel1
        vertical_weight2 = torch.zeros(self.out_channels, self.out_channels, 1, self.kernel_size, device=x.device)
        vertical_weight2[:, :, 0, :] = self.vertical_kernel2

        # 2. 水平卷积核 (9x1)
        horizontal_weight1 = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, 1, device=x.device)
        horizontal_weight1[:, :, :, 0] = self.horizontal_kernel1
        horizontal_weight2 = torch.zeros(self.out_channels, self.out_channels, self.kernel_size, 1, device=x.device)
        horizontal_weight2[:, :, :, 0] = self.horizontal_kernel2

        # 3. 主对角线卷积核 (9x9)
        diag1_weight1 = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, device=x.device)
        diag1_weight2 = torch.zeros(self.out_channels, self.out_channels, self.kernel_size, self.kernel_size, device=x.device)
        diag2_weight1 = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, device=x.device)
        diag2_weight2 = torch.zeros(self.out_channels, self.out_channels, self.kernel_size, self.kernel_size, device=x.device)
        for i in range(self.kernel_size):
            diag1_weight1[:, :, i, i] = self.diag1_kernel1[:, :, i]
            diag1_weight2[:, :, i, i] = self.diag1_kernel2[:, :, i]
            diag2_weight1[:, :, i, self.kernel_size-1 - i] = self.diag2_kernel1[:, :, i]  # 副对角线索引为 (i, 8-i)
            diag2_weight2[:, :, i, self.kernel_size-1 - i] = self.diag2_kernel2[:, :, i]


        # 分别应用卷积
        padding = self.kernel_size//2
        out_vertical = x1+F.conv2d(F.conv2d(x, vertical_weight1, padding=(0,padding)),vertical_weight2, padding=(0,padding))  # padding=(0,4) 保持高度不变
        out_horizontal =x1+ F.conv2d(F.conv2d(x, horizontal_weight1, padding=(padding,0)), horizontal_weight2, padding=(padding,0))  # padding=(4,0) 保持宽度不变
        out_diag1 = x1+F.conv2d(F.conv2d(x, diag1_weight1, padding=padding), diag1_weight2, padding=padding)  # padding=4 保持尺寸不变
        out_diag2 = x1+F.conv2d(F.conv2d(x, diag2_weight1, padding=padding), diag2_weight2, padding=padding)


        return torch.cat([out_vertical, out_horizontal, out_diag1, out_diag2],1)

class Chosenfuse(nn.Module):
    def __init__(self,in_channel1, in_channel2,r=2):
        self.in_channel = 0 if in_channel1==in_channel2 else 1
        super(Chosenfuse, self).__init__()
        if self.in_channel:
           self.block1 = nn.Sequential(
               nn.Conv2d(in_channel1,in_channel2,1),
               nn.LeakyReLU(0.2,True),
               LayerNorm2d(in_channel2)
           )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel2,in_channel2//r,1),
            nn.LeakyReLU(.1,True),
            nn.Conv2d(in_channel2//r,in_channel2//r,1),
            nn.Conv2d(in_channel2//r,in_channel2,1)
        )
        self.ghost1 = nn.ModuleList([
            nn.Conv2d(in_channel2,in_channel2//2, 3,1,1),
            nn.Conv2d(in_channel2,in_channel2//2,3,1,1)
            ])

        self.block2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel2,in_channel2, 3,1,1,groups=in_channel2),
                nn.Conv2d(in_channel2, in_channel2, 1)),
            nn.Sequential(
                nn.Conv2d(in_channel2, in_channel2, 3,1,1, groups=in_channel2),
                nn.Conv2d(in_channel2, in_channel2, 1)),
        ])


    def forward(self,x1,x2):
        if self.in_channel:
            x1 = self.block1(x1)
        a = x1
        b = x2
        x1 = a+a-b
        x2 = b+b-a

        fx1 = self.ghost1[0](x1)
        fx2 = self.ghost1[1](x2)
        w = torch.cat([fx1,fx2],1)
        w = (self.block2[0](w))+ (self.block2[1](w))

        g = self.mlp(self.gap(a+b))
        out = g*w+w

        return out


class LayerNorm2d(nn.Module):
    def __init__(self,num_channels,eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels,eps=eps)
        # self.norm = DyT(num_channels)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x=self.norm(x)
        x = x.permute(0,3,1,2)
        return x

class DyT(nn.Module):
    def __init__(self, dim, init_alpha=0.5):
        super(DyT, self).__init__()
        # 可学习的标量alpha，初始化为0.5（默认值）
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        # 逐通道的gamma和beta，与LN保持一致
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x的形状：[batch_size, num_tokens, dim]
        x = torch.tanh(self.alpha * x)  # 应用tanh和alpha缩放
        return self.gamma * x + self.beta  # 逐通道仿射变换

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                SpatialAttention(kernel_size=1,padding=0),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim+reduction_dim*len(bins), in_dim+reduction_dim*len(bins), kernel_size=3, padding=1,
                      bias=False,groups=in_dim+reduction_dim*len(bins)),
            nn.Conv2d(in_dim+reduction_dim*len(bins),in_dim,1),
            nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)*x