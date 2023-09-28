import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

BATCHNORM_TRACK_RUNNING_STATS = False
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


class BNorm_init(nn.BatchNorm2d):
    def reset_parameters(self):
        init.uniform_(self.weight, 0, 1)
        init.zeros_(self.bias)


class Conv2d_init(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super(Conv2d_init, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def reset_parameters(self):
        init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


def _conv_block(in_chanels, out_chanels, kernel_size, padding):
    return nn.Sequential(Conv2d_init(in_channels=in_chanels, out_channels=out_chanels,
                                     kernel_size=kernel_size, padding=padding, bias=False),
                         FeatureNorm(num_features=out_chanels, eps=0.001),
                         nn.ReLU())

class FeatureNorm(nn.Module):
    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(2, 3), eps=0.001, include_bias=True):
        super(FeatureNorm, self).__init__()
        self.shape = [1] * rank
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims

        self.scale = nn.Parameter(torch.ones(self.shape, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(self.shape, requires_grad=True, dtype=torch.float)) if include_bias else nn.Parameter(
            torch.zeros(self.shape, requires_grad=False, dtype=torch.float))

        self.eps = eps

    def forward(self, features):
        f_std = torch.std(features, dim=self.reduce_dims, keepdim=True)
        f_mean = torch.mean(features, dim=self.reduce_dims, keepdim=True)
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()) + self.bias

class UpSampling(nn.Module):
    """
    UpSampling blok - dekonvolucija(2x povečava resolucije) + konvolucija 
    """
    def __init__(self, n_conv_blocks, in_channels, out_channels, n_channels_connected, kernel_size, padding, stride=2):
        super(UpSampling, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.conv = nn.Sequential()

        for i in range(n_conv_blocks):
            if i == 0:
                self.conv.add_module(f'conv_block_{i+1}', _conv_block(n_channels_connected + out_channels, out_channels, kernel_size, padding))
            else:
                self.conv.add_module(f'conv_block_{i+1}', _conv_block(out_channels, out_channels, kernel_size, padding))

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Se_module_diff(nn.Module):
    def __init__(self, inp, oup, Avg_size = 1, se_ratio = 1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((Avg_size, Avg_size))
        num_squeezed_channels = max(1,int(inp / se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=inp, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        self.Avg_size = Avg_size
        self.reset_parameters()

    #x and z are different conv layer and z pass through more convs
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x, z):
        SIZE = z.size()
        y = self.avg(x)
        y = self._se_reduce(y)
        y = y * torch.sigmoid(y)
        y = self._se_expand(y)
        if self.Avg_size != 1:
            y = F.upsample_bilinear(y, size=[SIZE[2], SIZE[3]])
        z = torch.sigmoid(y) * z
        return z

class DownSampling(nn.Module):
    """
    DownSampling blok - na zacetku lahko dodamo MaxPooling
    """
    def __init__(self, pooling, n_conv_blocks, in_channels, out_channels, kernel_size, padding):
        super(DownSampling, self).__init__()
        self.downsample = nn.Sequential()

        if pooling:
            self.downsample.add_module('max_pooling', nn.MaxPool2d(2))

        for i in range(n_conv_blocks):
            self.downsample.add_module(f'conv_block_{i+1}', _conv_block(in_channels, out_channels, kernel_size, padding))
            in_channels = out_channels
    
    def forward(self, x):
        x = self.downsample(x)
        return x

class SegDecNetPlusPlus(nn.Module):
    def __init__(self, device, input_width, input_height, input_channels):
        super(SegDecNetPlusPlus, self).__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        
        self.volume1 = DownSampling(pooling=False, n_conv_blocks=1, in_channels=self.input_channels, out_channels=32, kernel_size=5, padding=2)
        self.volume2 = DownSampling(pooling=True, n_conv_blocks=3, in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.volume3 = DownSampling(pooling=True, n_conv_blocks=4, in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.volume4 = DownSampling(pooling=True, n_conv_blocks=1, in_channels=64, out_channels=1024, kernel_size=15, padding=7)

        self.extractor = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=1025, out_chanels=8, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=16, out_chanels=32, kernel_size=5, padding=2))

        self.global_max_pool_feat = nn.MaxPool2d(kernel_size=32)
        self.global_avg_pool_feat = nn.AvgPool2d(kernel_size=32)
        self.global_max_pool_seg = nn.MaxPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))
        self.global_avg_pool_seg = nn.AvgPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))

        self.fc = nn.Linear(in_features=66, out_features=1)

        # Custom autgrad funkcije - Gradient multiplyers
        self.volume_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer().apply

        self.device = device

        # Upsampling
        self.upsampling1 = UpSampling(n_conv_blocks=1, in_channels=1024, out_channels=16, n_channels_connected=64, kernel_size=5, padding=2)
        self.upsampling2 = UpSampling(n_conv_blocks=4, in_channels=16, out_channels=16, n_channels_connected=64, kernel_size=5, padding=2)
        self.upsampling3 = UpSampling(n_conv_blocks=3, in_channels=16, out_channels=8, n_channels_connected=32, kernel_size=5, padding=2)
        self.upsampling4 = nn.Sequential(Conv2d_init(in_channels=8, out_channels=1, kernel_size=5, padding=2, bias=False), FeatureNorm(num_features=1, eps=0.001, include_bias=False))

        # Downsampling
        self.downsampling = nn.AvgPool2d(8)
        
        # SSE Module
        self.conv1_s = nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False)
        self.conv2_s = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.conv3_s = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)

        self.se_module_diff1 = Se_module_diff(inp=32, oup=32)
        self.se_module_diff2 = Se_module_diff(inp=64, oup=64)
        self.se_module_diff2 = Se_module_diff(inp=64, oup=64)

    def set_gradient_multipliers(self, multiplier):
        self.volume_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_max_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_avg_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)

    def forward(self, input):
        v1 = self.volume1(input)
        v2 = self.volume2(v1)
        v3 = self.volume3(v2)
        v4 = self.volume4(v3)

        conv1_s = self.conv1_s(v1)
        conv1_sse = self.se_module_diff1(v1, conv1_s)

        conv2_s = self.conv2_s(v2)
        conv2_sse = self.se_module_diff2(v2, conv2_s)
        
        conv3_s = self.conv3_s(v3)
        conv3_sse = self.se_module_diff2(v3, conv3_s)

        seg_mask_upsampled = self.upsampling1(v4, conv3_sse)
        seg_mask_upsampled = self.upsampling2(seg_mask_upsampled, conv2_sse)
        seg_mask_upsampled = self.upsampling3(seg_mask_upsampled, conv1_sse)
        seg_mask_upsampled = self.upsampling4(seg_mask_upsampled)

        seg_mask_downsampled = self.downsampling(seg_mask_upsampled)

        cat = torch.cat([v4, seg_mask_downsampled], dim=1)

        cat = self.volume_lr_multiplier_layer(cat, self.volume_lr_multiplier_mask)

        features = self.extractor(cat)
        global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] # torch.Size([1, 32, 1, 1])
        global_avg_feat = torch.mean(features, dim=(-1, -2), keepdim=True) # torch.Size([1, 32, 1, 1])
        global_max_seg = torch.max(torch.max(seg_mask_upsampled, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] # torch.Size([1, 1, 1, 1])
        global_avg_seg = torch.mean(seg_mask_upsampled, dim=(-1, -2), keepdim=True) # torch.Size([1, 1, 1, 1])

        global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1) # torch.Size([1, 32])
        global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1) # torch.Size([1, 32])

        global_max_seg = global_max_seg.reshape(global_max_seg.size(0), -1) # torch.Size([1, 1])
        global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask) # (torch.Size([1, 1]), torch.Size([1])) -> torch.Size([1, 1])
        global_avg_seg = global_avg_seg.reshape(global_avg_seg.size(0), -1) # torch.Size([1, 1])
        global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask) # (torch.Size([1, 1]), torch.Size([1])) -> torch.Size([1, 1])

        fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg, global_avg_seg], dim=1) # torch.Size([1, 66])
        fc_in = fc_in.reshape(fc_in.size(0), -1) # torch.Size([1, 66])
        prediction = self.fc(fc_in) # torch.Size([1, 1])
        return prediction, seg_mask_upsampled


class SegDecNetOriginalJIM(nn.Module):
    def __init__(self, device, input_width, input_height, input_channels):
        super(SegDecNetOriginalJIM, self).__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.volume1 = DownSampling(pooling=False, n_conv_blocks=1, in_channels=self.input_channels, out_channels=32,
                                    kernel_size=5, padding=2)
        self.volume2 = DownSampling(pooling=True, n_conv_blocks=3, in_channels=32, out_channels=64, kernel_size=5,
                                    padding=2)
        self.volume3 = DownSampling(pooling=True, n_conv_blocks=4, in_channels=64, out_channels=64, kernel_size=5,
                                    padding=2)
        self.volume4 = DownSampling(pooling=True, n_conv_blocks=1, in_channels=64, out_channels=1024, kernel_size=15,
                                    padding=7)

        self.extractor = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=1025, out_chanels=8, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=16, out_chanels=32, kernel_size=5, padding=2))

        self.global_max_pool_feat = nn.MaxPool2d(kernel_size=32)
        self.global_avg_pool_feat = nn.AvgPool2d(kernel_size=32)
        self.global_max_pool_seg = nn.MaxPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))
        self.global_avg_pool_seg = nn.AvgPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))

        self.fc = nn.Linear(in_features=66, out_features=1)

        # Custom autgrad funkcije - Gradient multiplyers
        self.volume_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer().apply

        self.device = device

        self.seg_mask = nn.Sequential(
            Conv2d_init(in_channels=1024, out_channels=1, kernel_size=1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))

    def set_gradient_multipliers(self, multiplier):
        self.volume_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_max_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_avg_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)

    def forward(self, input):
        v1 = self.volume1(input)
        v2 = self.volume2(v1)
        v3 = self.volume3(v2)
        v4 = self.volume4(v3)

        seg_mask_downsampled = self.seg_mask(v4)
        seg_mask_upsampled = seg_mask_downsampled

        cat = torch.cat([v4, seg_mask_downsampled], dim=1)

        cat = self.volume_lr_multiplier_layer(cat, self.volume_lr_multiplier_mask)

        features = self.extractor(cat)
        global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_feat = torch.mean(features, dim=(-1, -2), keepdim=True)
        global_max_seg = torch.max(torch.max(seg_mask_upsampled, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_seg = torch.mean(seg_mask_upsampled, dim=(-1, -2), keepdim=True)

        global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1)
        global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1)

        global_max_seg = global_max_seg.reshape(global_max_seg.size(0), -1)
        global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
        global_avg_seg = global_avg_seg.reshape(global_avg_seg.size(0), -1)
        global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)

        fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg, global_avg_seg], dim=1)
        fc_in = fc_in.reshape(fc_in.size(0), -1)
        prediction = self.fc(fc_in)
        return prediction, F.interpolate(seg_mask_downsampled, scale_factor=8, mode="nearest")


class GradientMultiplyLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_bw):
        ctx.save_for_backward(mask_bw)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask_bw, = ctx.saved_tensors
        return grad_output.mul(mask_bw), None